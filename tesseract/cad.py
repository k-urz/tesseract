"""Experimental component coupons, not a completed tesseract assembly.

Parts are manufacturing-neutral BREP solids. Clearances, fasteners and tube
sections are nominal test dimensions. No printable-assembly/holding claim.
"""

import csv
import math
from pathlib import Path

import numpy as np

from .analysis import Envelope, telescope_requirements, write_json
from .config import Hardware


def engine():
    try:
        import build123d as b
    except ImportError as exc:
        raise ImportError('CAD requires the optional dependency: pip install -e ".[cad]"') from exc
    return b


def along(shape, origin, direction):
    b = engine()
    return b.Plane(origin=tuple(origin), z_dir=tuple(direction)).location * shape


def cylinder(radius, length, origin, direction):
    b = engine()
    return along(b.Cylinder(radius, length, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)), origin, direction)


def nut_pocket(h, origin, direction):
    b = engine()
    profile = b.RegularPolygon((h.nut_across_flats_mm + h.hole_clearance_mm) / math.sqrt(3), 6)
    return along(b.extrude(profile, amount=h.nut_thickness_mm + .5), origin, direction)


def tube_parts(h: Hardware, housing_length):
    b = engine()
    w, wall, end = h.tube_width_mm, h.tube_wall_mm, h.end_wall_mm
    overlap = h.min_overlap_mm
    diameter = h.bolt_diameter_mm + h.hole_clearance_mm
    travel = housing_length-end-overlap
    if travel < diameter+4:
        raise ValueError("housing too short for overlap and stop-pin end ligaments")
    outer = b.Box(w, w, housing_length, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    bore = b.Pos(0, 0, end) * b.Box(w-2*wall, w-2*wall, housing_length, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    outer -= bore
    # Two transverse slots and a removable screw through the inner rod form
    # a positive end stop. The cap closes the other end of the outer tube.
    low, high = end+overlap/2, housing_length-overlap/2
    slot = b.Pos(0, 0, (low+high)/2) * b.Box(w*3, diameter, high-low)
    slot += cylinder(diameter/2, w*3, (-w*1.5, 0, low), (1, 0, 0))
    slot += cylinder(diameter/2, w*3, (-w*1.5, 0, high), (1, 0, 0))
    outer -= slot
    boss_width = max(w+2, h.nut_across_flats_mm+4)
    boss_depth = h.nut_thickness_mm+4
    boss_z = housing_length-6
    boss_y = w/2 + boss_depth/2 - .5
    outer += b.Pos(0, boss_y, boss_z) * b.Box(boss_width, boss_depth, 10)
    outer -= cylinder(diameter/2, boss_depth+wall+3, (0, w/2-wall-.5, boss_z), (0, 1, 0))
    pocket_start = w/2 + boss_depth - .5 - h.nut_thickness_mm
    outer -= nut_pocket(h, (0, pocket_start, boss_z), (0, 1, 0))
    inner = b.Box(h.inner_width_mm, h.inner_width_mm, housing_length-end,
                  align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    inner -= cylinder(diameter/2, w*3, (-w*1.5, 0, overlap/2), (1, 0, 0))
    return {"telescopic_outer": outer, "telescopic_inner": inner}


def ball_parts(h: Hardware):
    b = engine()
    r = h.ball_radius_mm
    half_x, half_y, half_z = r+5, r+7.5, r+1.5
    housing = b.Box(2*half_x, 2*half_y, 2*half_z)
    housing -= b.Sphere(r+h.socket_clearance_mm)
    opening = max(h.neck_radius_mm+.5, r*.65+h.socket_clearance_mm)
    housing -= cylinder(opening, half_z+2, (0, 0, 0), (0, 0, 1))
    for y in (-r-3.5, r+3.5):
        housing -= cylinder((h.bolt_diameter_mm+h.hole_clearance_mm)/2, 2*half_x+4,
                            (-half_x-2, y, 0), (1, 0, 0))
        housing -= nut_pocket(h, (half_x-h.nut_thickness_mm, y, 0), (1, 0, 0))
    split = 2*h.socket_clearance_mm+.4
    left = housing & (b.Pos(-(half_x+split/2)/2, 0, 0) * b.Box(half_x-split/2, 2*half_y+2, 2*half_z+2))
    right = housing & (b.Pos((half_x+split/2)/2, 0, 0) * b.Box(half_x-split/2, 2*half_y+2, 2*half_z+2))
    ball = b.Sphere(r) + cylinder(h.neck_radius_mm, r+10-r/2, (0, 0, r/2), (0, 0, 1))
    return {"ball_stem": ball, "socket_left": left, "socket_right": right}


def fit_coupon(h, clearance):
    b = engine()
    w = h.tube_width_mm
    block = b.Box(w+8, w+8, 8, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    # Clearance per SIDE around the same inner square rod used by the sample.
    side = h.inner_width_mm + 2*clearance
    block -= b.Pos(0, 0, -1) * b.Box(side, side, 10, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    return block


def hub_gauge(study):
    b = engine()
    h = study.config.hardware
    scan = study.scans["prototype_candidate"]
    i = int(np.argmin(abs(scan["angles_deg"])))
    p = scan["points_mm"][i]
    if not np.isfinite(p).all():
        raise ValueError("hub gauge reference pose has invalid projection")
    hub = b.Sphere(h.hub_radius_mm)
    vertex = 0
    directions = []
    for edge_id, (a, end) in enumerate(study.edges):
        if vertex not in (a, end):
            continue
        other = end if a == vertex else a
        direction = p[other]-p[vertex]
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise ValueError("hub gauge reference contains a zero-length edge")
        direction = direction/norm
        directions.append({"edge_id": edge_id, "unit_direction": direction.tolist()})
        # Blind square seats at the four nominal edge directions. This rigid
        # gauge checks packaging only; it is not a four-way moving joint.
        seat = b.Box(h.tube_width_mm+2*h.slide_clearance_mm,
                     h.tube_width_mm+2*h.slide_clearance_mm, 10,
                     align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
        hub -= along(seat, direction*(h.hub_radius_mm-6), direction)
    return hub, {"vertex_id": vertex, "reference_angle_deg": float(scan["angles_deg"][i]), "directions": directions}


def required_tilts(study):
    scan = study.scans["prototype_candidate"]
    reference = scan["points_mm"][int(np.argmin(abs(scan["angles_deg"])))]
    rows = []
    for e, (a, b) in enumerate(study.edges):
        ref = reference[b]-reference[a]
        refnorm = np.linalg.norm(ref)
        vectors = scan["points_mm"][:, b]-scan["points_mm"][:, a]
        norms = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(vectors).all(axis=1) & (norms > 1e-8)
        if refnorm > 1e-8 and valid.any():
            cos = vectors[valid] @ ref / (norms[valid]*refnorm)
            tilt = float(np.rad2deg(np.arccos(np.clip(cos, -1, 1))).max())
        else:
            tilt = None
        rows.append({"edge_id": e, "max_tilt_from_reference_deg": tilt})
    return rows


def make_parts(study):
    h = study.config.hardware
    requirements = telescope_requirements(study, "prototype_candidate")
    feasible = [row for row in requirements if row["two_stage_fits"]]
    sample = max(feasible, key=lambda r: (r["ratio"], r["required_stroke_mm"])) if feasible else None
    # Still provide a bench coupon if no edge can use a two-stage tube, but
    # explicitly mark it as unassigned, not fitted to a target edge.
    length = sample["housing_length_mm"] if sample else max(80, h.min_overlap_mm+h.end_wall_mm+20)
    parts = {**tube_parts(h, length), **ball_parts(h)}
    gauge, gauge_meta = hub_gauge(study)
    parts["hub_layout_gauge"] = gauge
    for i, clearance in enumerate(h.fit_clearances_mm):
        parts[f"fit_coupon_{i+1}_{clearance:g}mm"] = fit_coupon(h, clearance)
    for name, shape in parts.items():
        if not shape.is_valid or shape.volume <= 0 or len(shape.solids()) != 1:
            raise ValueError(f"invalid, empty or disconnected CAD part: {name}")
    return parts, sample, length, gauge_meta


def preview(parts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    names = ["telescopic_outer", "telescopic_inner", "ball_stem", "socket_left", "socket_right", "hub_layout_gauge"]
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    for i, name in enumerate(names):
        shape = parts[name]
        ax = fig.add_subplot(2, 3, i+1, projection="3d")
        vertices, triangles = shape.tessellate(.7)
        xyz = np.array([tuple(v) for v in vertices])
        faces = xyz[np.array(triangles)]
        ax.add_collection3d(Poly3DCollection(faces, facecolor="#58b3ba", edgecolor="#235565", linewidth=.15, alpha=.95))
        center = (xyz.min(axis=0)+xyz.max(axis=0))/2
        radius = np.ptp(xyz, axis=0).max()*.55
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            setter(c-radius, c+radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Experimental components / independent fit and packaging tests")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def export_cad(study, output: Path):
    b = engine()
    parts, sample, length, gauge_meta = make_parts(study)
    h = study.config.hardware
    output.mkdir(parents=True, exist_ok=True)
    part_rows = []
    for name, shape in parts.items():
        b.export_step(shape, output / f"{name}.step")
        if not b.export_stl(shape, output / f"{name}.stl", tolerance=.08, angular_tolerance=.15):
            raise ValueError(f"STL export failed: {name}")
        bounds = shape.bounding_box()
        part_rows.append({"part_id": name, "quantity": 1, "volume_mm3": shape.volume,
                          "x_mm": bounds.size.X, "y_mm": bounds.size.Y, "z_mm": bounds.size.Z,
                          "status": "experimental; material/process/fit unvalidated"})
    preview(parts, output / "components.png")
    tilts = []
    housing = parts["socket_left"] + parts["socket_right"]
    for angle in range(0, 66, 5):
        ball = b.Rot(0, angle, 0) * parts["ball_stem"]
        common = ball & housing
        volume = common.volume if common is not None else 0
        tilts.append({"tilt_deg": angle, "intersection_volume_mm3": volume, "clear": volume < 1e-5})
    # Conservative second pass including the local friction-boss radial extent
    # and an allowance for screw heads. This is deliberately pessimistic along
    # the entire rod; a flagged result must not be represented as a safe design.
    bounds = parts["telescopic_outer"].bounding_box()
    rod_radius = math.hypot(max(abs(bounds.min.X), abs(bounds.max.X)), max(abs(bounds.min.Y), abs(bounds.max.Y))) + h.bolt_diameter_mm
    socket_radius = math.sqrt((h.ball_radius_mm+5)**2+(h.ball_radius_mm+7.5)**2+(h.ball_radius_mm+1.5)**2)
    hub_radius = max(h.hub_radius_mm, socket_radius)
    envelope = Envelope(study.edges)
    recheck = {}
    for name in ("prototype_candidate", "target_candidate"):
        checks = [envelope.check(p, hub_radius, rod_radius) for p in study.scans[name]["points_mm"]]
        margins = [c["minimum_clearance_mm"] for c in checks if c["minimum_clearance_mm"] is not None]
        recheck[name] = {"blocked_samples": sum(c["blocked"] for c in checks), "samples": len(checks),
                         "min_clearance_mm": min(margins) if margins else None}
    manifest = {"config_id": study.config.digest(), "units": "mm", "manufacturing_process": "unselected",
                "parts": part_rows, "sample_edge": sample, "housing_length_mm": length, "hub_gauge": gauge_meta,
                "required_edge_direction_changes": required_tilts(study), "socket_tilt_samples": tilts,
                "direction_change_interpretation": "World-frame direction changes relative to the reference pose, assuming a non-rotating hub frame. A real moving hub needs its own orientation model.",
                "post_cad_envelope": {"hub_radius_mm": hub_radius, "rod_radius_mm": rod_radius, "scans": recheck,
                    "interpretation": "Conservative entire-rod envelope including a friction boss and screw allowance. A collision flags redesign, not an exact localized collision."},
                "all_edge_requirements": {name: telescope_requirements(study, name) for name in ("prototype_candidate", "target_candidate")},
                "limitations": ["Independent component trials only; ball stems have no final rod adapters.",
                                "The rigid hub is a layout gauge, not an articulated four-edge hub.",
                                "Tilt tested about one axis in 5 degree increments; not a full angular workspace test.",
                                "No material, load, friction, holding or fatigue validation.",
                                "Do not order a full set of 32 rods from these trial files."]}
    write_json(output / "manifest.json", manifest)
    with (output / "parts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(part_rows[0])); writer.writeheader(); writer.writerows(part_rows)
    hardware = [
        {"item": "stop pin / screw with washers and nut", "quantity": 1, "nominal_diameter_mm": h.bolt_diameter_mm, "length": "measure assembled tube width plus washers/nut; confirm head clearance"},
        {"item": "friction adjustment screw and nut", "quantity": 1, "nominal_diameter_mm": h.bolt_diameter_mm, "length": "measure boss-to-inner-rod distance; prevent over-travel into the rod"},
        {"item": "socket clamp screws and nuts", "quantity": 2, "nominal_diameter_mm": h.bolt_diameter_mm, "length": "housing width plus head/nut/washer allowance"}]
    write_json(output / "hardware.json", hardware)
    edge_label = f"E{sample['edge_id']}" if sample else "unassigned bench coupon"
    instructions = f"""# Experimental component pack

Config `{study.config.digest()}`; all dimensions mm. Process/material unselected.

## What this pack establishes

- A two-stage tube test assigned to **{edge_label}**; housing length {length:.3f} mm.
- Removable transverse stop pin, closed outer end and separate friction adjustment screw.
- Split socket/ball test and a rigid four-direction layout gauge at V0.
- Per-side clearance coupons labelled by filename. Use the supplied inner rod to compare fits.

## Assembly and measurements

1. Inspect STEP dimensions, STL units, wall thickness and the chosen supplier's process limits.
2. Make one fit coupon and one short rod first. Record process, material, orientation and finish.
3. Deburr/finish the bore as appropriate; insert the inner rod BEFORE installing its transverse stop pin.
4. Install the stop pin through the slots and inner cross-hole; use washers and a nut. Check the two end stops and retained overlap.
5. Fit the friction screw in its separate boss. The screw presses the inner rod; do not cut printed threads by assumption.
6. Place the ball between the socket halves and install two screws/nuts. Tighten incrementally; check freedom, retention and material deformation.
7. Record the force to start/continue motion, holding force, backlash, available stroke and socket angle before and after repeated use.
8. Use the hub as a static placement gauge only. Design the articulated hub and rod adapters after these measurements.

`manifest.json` records failed edge packaging checks, required direction changes, sampled stem/socket intersections and the larger post-CAD envelopes. They remain design blockers until resolved. The pack is NOT a complete tesseract assembly.
"""
    (output / "ASSEMBLY.md").write_text(instructions, encoding="utf-8")
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part_id", "config_id", "process", "material", "orientation", "finish", "measured_clearance_mm", "stroke_mm", "start_force_N", "moving_force_N", "holding_force_N", "backlash_mm", "max_tilt_deg", "cycles", "notes"])
        for part in part_rows:
            writer.writerow([part["part_id"], study.config.digest()] + [""]*13)
