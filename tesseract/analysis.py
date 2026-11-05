"""Sampled envelope checks and reproducible reports (not a physics solver)."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import csv
import json
import platform
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from . import __version__
from .config import Config
from .geometry import (EPS, corner_angles, corner_pairs, hypercube,
                       point_segment_distance, project, rotate, samples, segment_distance)


def clean(value):
    """Produce strict JSON; undefined geometry is null, never NaN or Infinity."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Envelope:
    """Spherical hubs and square-rod circumscribed capsules outside each hub.

    Incident rods are checked AFTER trimming their shared hub region. Nominal
    joint overlap inside the hub is intentional, not a reason to skip the whole
    pair. The ball-clamp envelope is checked separately when CAD is generated.
    """

    def __init__(self, edges):
        self.edges = edges
        self.vv = np.array(list(combinations(range(16), 2)))
        self.ee = np.array(list(combinations(range(len(edges)), 2)))
        self.ve = np.array([(v, e) for v in range(16) for e, ends in enumerate(edges) if v not in ends])

    def check(self, points, hub_radius, rod_radius):
        if not np.isfinite(points).all():
            return {"projection_invalid": True, "blocked": True, "minimum_clearance_mm": None,
                    "clearances_mm": {}, "colliding_edges": [], "colliding_vertices": [], "critical_pairs": [],
                    "invalid_projection_vertices": np.flatnonzero(~np.isfinite(points).all(axis=1)).tolist(), "zero_length_edges": []}
        a, b = points[self.edges[:, 0]], points[self.edges[:, 1]]
        lengths = np.linalg.norm(b - a, axis=1)
        unit = np.zeros_like(a)
        np.divide(b-a, lengths[:, None], out=unit, where=lengths[:, None] > EPS)
        trim = np.minimum(hub_radius, lengths / 2)
        start, end = a + trim[:, None] * unit, b - trim[:, None] * unit
        vv = np.linalg.norm(points[self.vv[:, 0]] - points[self.vv[:, 1]], axis=1) - 2 * hub_radius
        ee = segment_distance(start[self.ee[:, 0]], end[self.ee[:, 0]], start[self.ee[:, 1]], end[self.ee[:, 1]]) - 2 * rod_radius
        ve = point_segment_distance(points[self.ve[:, 0]], start[self.ve[:, 1]], end[self.ve[:, 1]]) - hub_radius - rod_radius
        spans = lengths - 2 * hub_radius
        bad_edges = set(np.flatnonzero(spans <= EPS).tolist())
        bad_vertices = set()
        for pair in self.ee[ee <= EPS]:
            bad_edges.update(pair.tolist())
        for vertex, edge in self.ve[ve <= EPS]:
            bad_vertices.add(int(vertex)); bad_edges.add(int(edge))
        for pair in self.vv[vv <= EPS]:
            bad_vertices.update(pair.tolist())
        groups = (("hub_hub", vv, self.vv, "vertices"),
                  ("rod_rod_outside_hubs", ee, self.ee, "edges"),
                  ("hub_rod", ve, self.ve, "vertex_edge"))
        critical = []
        for kind, distances, pairs, ids in groups:
            i = int(np.argmin(distances))
            critical.append({"kind": kind, ids: pairs[i].tolist(), "clearance_mm": float(distances[i])})
        margins = {kind: float(values.min()) for kind, values, _, _ in groups}
        margins["edge_space_between_hubs"] = float(spans.min())
        return {"projection_invalid": False, "blocked": min(margins.values()) <= EPS,
                "minimum_clearance_mm": min(margins.values()), "clearances_mm": margins,
                "colliding_edges": sorted(bad_edges), "colliding_vertices": sorted(bad_vertices),
                "critical_pairs": critical, "invalid_projection_vertices": [],
                "zero_length_edges": np.flatnonzero(lengths <= EPS).tolist()}


def finite_range(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else (None, None)


def clear_intervals(angles, checks):
    """Runs of clear SAMPLES, not certified continuously clear intervals."""
    runs = []
    start = previous = None
    for angle, check in zip(angles, checks):
        if not check["blocked"]:
            if start is None:
                start = float(angle)
            previous = float(angle)
        elif start is not None:
            runs.append([start, previous]); start = previous = None
    if start is not None:
        runs.append([start, previous])
    return runs


def summarize(angles, points, edges, corners, envelope, config, size_mm, name, scale):
    lengths = np.linalg.norm(points[:, edges[:, 0]] - points[:, edges[:, 1]], axis=-1)
    joints = np.array([corner_angles(p, corners) for p in points])
    checks = [envelope.check(p, config.hardware.hub_radius_mm, config.hardware.rod_radius_mm) for p in points]
    edge_rows = []
    for e, (a, b) in enumerate(edges):
        lo, hi = finite_range(lengths[:, e])
        valid = np.flatnonzero(np.isfinite(lengths[:, e]))
        def extreme(which):
            return float(angles[valid[which(lengths[valid, e])]]) if valid.size else None
        edge_rows.append({"edge_id": e, "vertex_a": int(a), "vertex_b": int(b), "min_mm": lo,
                          "max_mm": hi, "stroke_mm": hi-lo if lo is not None else None,
                          "ratio": hi/lo if lo is not None and lo > EPS else None,
                          "min_at_deg": extreme(np.argmin), "max_at_deg": extreme(np.argmax)})
    corner_rows = []
    for i, row in enumerate(corners):
        lo, hi = finite_range(joints[:, i])
        corner_rows.append({"vertex_id": int(row[0]), "edge_a": int(row[1]), "edge_b": int(row[2]),
                            "min_deg": lo, "max_deg": hi})
    clear = [c["minimum_clearance_mm"] for c in checks if c["minimum_clearance_mm"] is not None]
    bbox = [float(np.ptp(p, axis=0).max() + 2*config.hardware.hub_radius_mm) for p in points if np.isfinite(p).all()]
    summary = {"name": name, "size_mm": size_mm, "scale_mm_per_unit": scale,
               "samples": len(angles), "start_deg": float(angles[0]), "end_deg": float(angles[-1]),
               "invalid_projection_samples": sum(c["projection_invalid"] for c in checks),
               "undefined_corner_angles": int(np.count_nonzero(~np.isfinite(joints))),
               "zero_edge_samples": int(np.count_nonzero(np.any(lengths <= EPS, axis=1))),
               "blocked_samples": sum(c["blocked"] for c in checks),
               "sampled_envelope_clear": not any(c["blocked"] for c in checks),
               "clear_sample_intervals_deg": clear_intervals(angles, checks),
               "min_clearance_mm": min(clear) if clear else None,
               "max_sampled_envelope_mm": max(bbox) if bbox else None,
               "edge_length_range_mm": finite_range(lengths), "joint_angle_range_deg": finite_range(joints)}
    return {"summary": summary, "angles_deg": angles, "points_mm": points, "lengths_mm": lengths,
            "joint_angles_deg": joints, "edges": edge_rows, "corners": corner_rows, "checks": checks}


@dataclass
class Study:
    config: Config
    vertices: np.ndarray
    edges: np.ndarray
    corners: np.ndarray
    scans: dict
    convergence: dict


def analyze(config: Config) -> Study:
    config.validate()
    vertices, edges = hypercube()
    corners, envelope = corner_pairs(edges), Envelope(edges)
    m = config.model
    grids = {"full": samples(0, 360, m.coarse_step_deg),
             "candidate_coarse": samples(m.candidate_start_deg, m.candidate_end_deg, m.coarse_step_deg),
             "candidate": samples(m.candidate_start_deg, m.candidate_end_deg, m.fine_step_deg)}
    projected = {name: np.array([project(rotate(vertices, a, m.plane), m.projection, m.projection_distance) for a in grid]) for name, grid in grids.items()}
    extents = [np.ptp(p, axis=0).max() for poses in projected.values() for p in poses if np.isfinite(p).all()]
    if not extents or max(extents) <= EPS:
        raise ValueError("no finite, nonzero projected pose; adjust projection distance or plane")
    # One fixed scale for the WHOLE motion, with a small sampled-size reserve.
    normalized_extent = max(extents) * 1.005
    scans = {}
    for prefix, size in (("prototype", m.size_mm), ("target", m.target_size_mm)):
        scale = (size - 2 * config.hardware.hub_radius_mm) / normalized_extent
        for name, angles in grids.items():
            scans[f"{prefix}_{name}"] = summarize(angles, projected[name]*scale, edges, corners, envelope,
                                                  config, size, f"{prefix}_{name}", scale)
    coarse = scans["prototype_candidate_coarse"]
    fine = scans["prototype_candidate"]
    changes = []
    for a, b in zip(coarse["edges"], fine["edges"]):
        if a["min_mm"] is not None and b["min_mm"] is not None:
            changes += [abs(a["min_mm"]-b["min_mm"]), abs(a["max_mm"]-b["max_mm"])]
    ca, cb = coarse["summary"]["min_clearance_mm"], fine["summary"]["min_clearance_mm"]
    convergence = {"coarse_step_deg": m.coarse_step_deg, "fine_step_deg": m.fine_step_deg,
                   "max_edge_extremum_change_mm": max(changes) if changes else None,
                   "min_clearance_change_mm": abs(ca-cb) if ca is not None and cb is not None else None,
                   "limitation": "Discrete samples only; convergence is not a continuous collision-free certificate."}
    joint_changes = [abs(a[key]-b[key]) for a,b in zip(coarse["corners"],fine["corners"])
                     for key in ("min_deg","max_deg") if a[key] is not None and b[key] is not None]
    convergence["max_joint_extremum_change_deg"] = max(joint_changes) if joint_changes else None
    return Study(config, vertices, edges, corners, scans, convergence)


def telescope_requirements(study, scan_name):
    """Two-stage square rod, with end wall and minimum retained overlap.

    Each end reserves one hub radius. A transverse stop pin at overlap/2 from
    the inner rod's tail travels in a slot; a separate screw supplies friction.
    """
    h = study.config.hardware
    rows = []
    for edge in study.scans[scan_name]["edges"]:
        minimum, maximum = edge["min_mm"], edge["max_mm"]
        housing = minimum - 2*h.hub_radius_mm if minimum is not None else None
        capacity = housing - h.end_wall_mm - h.min_overlap_mm if housing is not None else None
        required = edge["stroke_mm"]
        pin_diameter = h.bolt_diameter_mm + h.hole_clearance_mm
        # Leave end ligaments around the slot and room for a friction boss.
        min_housing = h.end_wall_mm + h.min_overlap_mm + pin_diameter + 4
        ok = housing is not None and housing >= min_housing and required <= capacity + 1e-8
        rows.append({**edge, "housing_length_mm": housing, "inner_rod_length_mm": housing-h.end_wall_mm if housing is not None else None,
                     "capacity_mm": capacity, "required_stroke_mm": required,
                     "stroke_shortfall_mm": max(0, required-capacity) if required is not None else None,
                     "minimum_housing_length_mm": min_housing, "two_stage_fits": ok})
    return rows


def export_analysis(study: Study, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    dependencies = {}
    for name in ("numpy", "matplotlib", "build123d", "ocp-gordon", "cadquery-ocp"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    report = {"version": __version__, "python": platform.python_version(), "dependencies": dependencies, "units": {"length": "mm", "angle": "degree"},
              "config_id": study.config.digest(), "config": study.config.data(),
              "vertices_4d": study.vertices, "edge_topology": study.edges,
              "convergence": study.convergence, "scans": {},
              "limitations": ["Spherical hubs and capsule rod envelopes; no masses, elasticity or friction.",
                              "Joint angles are pairwise edge angles, not a ball-joint travel specification.",
                              "Full loop sampled coarsely; candidate sampled finely. No continuous clearance guarantee.",
                              "Nominal rod envelope excludes local friction bosses; CAD recheck uses a larger conservative radius."]}
    for name, scan in study.scans.items():
        rows = telescope_requirements(study, name)
        report["scans"][name] = {"summary": scan["summary"], "edges": scan["edges"], "corners": scan["corners"], "telescopes": rows,
                                 "frames": [{"angle_deg": a, "points_mm": p, **check} for a, p, check in zip(scan["angles_deg"], scan["points_mm"], scan["checks"])]}
        for suffix, data in (("edges", rows), ("corners", scan["corners"])):
            with (output / f"{name}_{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(clean(data))
    write_json(output / "analysis.json", report)
    write_json(output / "config.json", {"config_id": study.config.digest(), "config": study.config.data()})
    lines = ["# Sampled geometry study", "", f"Configuration: `{study.config.digest()}`. Lengths: mm; angles: degrees.", "",
             "| Scan | Samples | Edge range | Min clearance | Blocked | Two-stage edges fitting |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for name, scan in study.scans.items():
        s = scan["summary"]
        lo, hi = s["edge_length_range_mm"]
        fmt = lambda n: "undefined" if n is None else f"{n:.3f}"
        lines.append(f"| {name} | {s['samples']} | {fmt(lo)}–{fmt(hi)} | {fmt(s['min_clearance_mm'])} | {s['blocked_samples']} | {sum(r['two_stage_fits'] for r in telescope_requirements(study, name))}/32 |")
    lines += ["", "## Interpretation", "", "- Negative clearance means overlapping conservative envelopes at a sampled pose.",
              "- Rods sharing a hub are checked outside that hub, not skipped.",
              "- Two-stage fit is a packaging/stroke check, not proof of joint travel or strength.",
              "- A candidate can fail. Report failures before designing or ordering a complete model.",
              "- Hardware dimensions stay fixed between the two model sizes; miniaturization does not assume infinitely small hardware.",
              "- Sampling cannot certify absence of a collision between frames. No continuous-path, friction or holding claim is made.",
              "", "## Coarse/fine comparison", "", "```json", json.dumps(clean(study.convergence), indent=2), "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
