"""Local mobility and an exact four-square reduction of perspective xw motion.

These are point-geometry constraints, not a rigid-body mechanism or force model.
Ranks describe infinitesimal motion only; contacts and manufacturing are excluded.
"""

import csv
from pathlib import Path

import numpy as np

from .analysis import write_json
from .geometry import corner_angles, hypercube, project, rotate


# Stable order: source x sign varies fastest, then source w sign.
SIGNS = ((-1, -1), (1, -1), (-1, 1), (1, 1))
RANK_TOL = 1e-8


def edge_groups(vertices, edges):
    """Eight length families dictated by xw symmetry, not fit to sampled data."""
    groups = {}
    for edge_id, (a, b) in enumerate(edges):
        axis = (int(a) ^ int(b)).bit_length() - 1
        x, w = int(vertices[a, 0]), int(vertices[a, 3])
        name = f"square_x{x:+d}_w{w:+d}" if axis in (1, 2) else (
            f"x_link_w{w:+d}" if axis == 0 else f"w_link_x{x:+d}")
        groups.setdefault(name, []).append(edge_id)
    return groups


def jacobians(points, edges, corners):
    """Analytic derivatives of lengths and corner cosines w.r.t. 48 coordinates."""
    lengths = np.zeros((len(edges), points.size))
    cosines = np.zeros((len(corners), points.size))
    for i, (a, b) in enumerate(edges):
        delta = points[b] - points[a]
        length = np.linalg.norm(delta)
        if length < 1e-10:
            raise ValueError("constraint Jacobian undefined at zero-length edge")
        lengths[i, 3*a:3*a+3] = -delta / length
        lengths[i, 3*b:3*b+3] = delta / length
    for i, (a, _, _, b, c) in enumerate(corners):
        u, v = points[b] - points[a], points[c] - points[a]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        cosine = u @ v / (lu * lv)
        gu = v / (lu * lv) - cosine * u / lu**2
        gv = u / (lu * lv) - cosine * v / lv**2
        cosines[i, 3*b:3*b+3] = gu
        cosines[i, 3*c:3*c+3] = gv
        cosines[i, 3*a:3*a+3] = -gu - gv
    return lengths, cosines


def rank(matrix):
    """Normalize nonzero rows so lengths and cosines have comparable rank units."""
    matrix = np.asarray(matrix, float)
    norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix[norms > 1e-12] / norms[norms > 1e-12, None]
    return int(np.linalg.matrix_rank(matrix, tol=RANK_TOL)) if len(matrix) else 0


def independent_angles(length_jacobian, angle_jacobian):
    """One local complement, NOT a globally minimal physical joint design."""
    selected = []
    matrix = length_jacobian
    current = rank(matrix)
    for i, row in enumerate(angle_jacobian):
        candidate = np.vstack((matrix, row))
        new_rank = rank(candidate)
        if new_rank > current:
            selected.append(i)
            matrix, current = candidate, new_rank
        if current == 42:
            break
    return selected


def square_coordinates(angle_deg, distance=4.0, scale=1.0):
    """Four pairs (axial center X, square half-width B), in the requested scale."""
    if distance <= np.sqrt(2) or scale <= 0:
        raise ValueError("four-square model needs distance > sqrt(2) and scale > 0")
    theta = np.deg2rad(angle_deg)
    result = []
    for x, w in SIGNS:
        xr = x*np.cos(theta) - w*np.sin(theta)
        wr = x*np.sin(theta) + w*np.cos(theta)
        half_width = scale*distance / (distance - wr)
        result.append((half_width*xr, half_width))
    return np.array(result)


def square_vertices(coordinates):
    vertices, _ = hypercube()
    points = np.empty((16, 3))
    for i, (x, y, z, w) in enumerate(vertices):
        center, half_width = coordinates[SIGNS.index((int(x), int(w)))]
        points[i] = (center, y*half_width, z*half_width)
    return points


def square_constraints(coordinates, distance=4.0, scale=1.0):
    """Four conic equations + three phase equalities, and branch dot products.

    Equality residuals vanish for both same and opposite phase. Positive branch
    dot products MUST also hold to select the intended assembly branch.
    """
    coordinates = np.asarray(coordinates, float).reshape(4, 2)
    centers, half_widths = coordinates.T
    if np.any(half_widths <= 0):
        raise ValueError("square half-width must be positive")
    xr = centers / half_widths
    wr = distance * (1 - scale / half_widths)
    phase = np.array([((x*a+w*b)/2, (x*b-w*a)/2)
                      for (x, w), a, b in zip(SIGNS, xr, wr)])
    norms = np.sum(phase**2, axis=1) - 1
    crosses = phase[0, 0]*phase[1:, 1] - phase[0, 1]*phase[1:, 0]
    dots = phase[1:] @ phase[0]
    return np.concatenate((norms, crosses)), dots


def numerical_jacobian(function, q, step=1e-6):
    q = np.asarray(q, float).ravel()
    columns = []
    for i in range(len(q)):
        offset = np.zeros_like(q)
        offset[i] = step
        columns.append((function(q+offset) - function(q-offset)) / (2*step))
    return np.column_stack(columns)


def rigid_basis(points):
    columns = [np.tile(axis, (len(points), 1)).ravel() for axis in np.eye(3)]
    columns += [np.cross(axis, points).ravel() for axis in np.eye(3)]
    u, s, _ = np.linalg.svd(np.column_stack(columns), full_matrices=False)
    return u[:, s > 1e-10]


def intrinsic_rotation_dof(vertices, distance):
    """Six SO(4) directions, modulo translation/rotation of the whole 3D model."""
    p = project(vertices, distance=distance)
    rigid = rigid_basis(p)
    directions = np.column_stack([
        ((project(rotate(vertices, 1e-4, plane), distance=distance) -
          project(rotate(vertices, -1e-4, plane), distance=distance)) / np.deg2rad(2e-4)).ravel()
        for plane in ("xy", "xz", "xw", "yz", "yw", "zw")])
    residual = directions - rigid @ (rigid.T @ directions)
    return int(np.linalg.matrix_rank(residual, tol=RANK_TOL))


def range_counterexample(study):
    """An exact uniform shrink preserves angles/equal lengths, yet leaves the path."""
    scan = study.scans["prototype_candidate"]
    lows = np.array([e["min_mm"] for e in scan["edges"]])
    highs = np.array([e["max_mm"] for e in scan["edges"]])
    # Choose a pose strictly inside every length interval, if one is sampled.
    lengths = scan["lengths_mm"]
    margins = np.min((lengths-lows) / lengths, axis=1)
    index = int(np.argmax(margins))
    if margins[index] < 1e-9:
        return {"available": False, "reason": "No sampled pose strictly inside every edge interval."}
    factor = 1 - min(.001, margins[index]/2)
    points = scan["points_mm"][index]
    shrunk = factor*points
    new_lengths = np.linalg.norm(shrunk[study.edges[:, 0]] - shrunk[study.edges[:, 1]], axis=1)
    angle_error = float(np.max(abs(corner_angles(shrunk, study.corners) - corner_angles(points, study.corners))))
    m = study.config.model
    q = square_coordinates(scan["angles_deg"][index], m.projection_distance,
                           scan["summary"]["scale_mm_per_unit"])
    residual, _ = square_constraints(factor*q, m.projection_distance,
                                    scan["summary"]["scale_mm_per_unit"])
    return {"available": True, "angle_deg": float(scan["angles_deg"][index]), "shrink_factor": factor,
            "all_length_limits_satisfied": bool(np.all((new_lengths >= lows-1e-8) & (new_lengths <= highs+1e-8))),
            "max_corner_angle_change_deg": angle_error,
            "equal_length_families_preserved": True,
            "fixed_scale_path_constraint_error": float(np.max(abs(residual))),
            "interpretation": "Exact algebraic counterexample to limits/equalities selecting the fixed-scale path; not a hardware collision check."}


def constraint_study(study):
    m = study.config.model
    if m.plane != "xw" or m.projection != "perspective" or m.projection_distance <= np.sqrt(2):
        raise ValueError("constraints study currently requires xw perspective with distance > sqrt(2)")
    vertices, edges, corners = study.vertices, study.edges, study.corners
    groups = edge_groups(vertices, edges)
    yz_corners = [i for i, (_, ea, eb, _, _) in enumerate(corners)
                  if {int(edges[ea, 0]) ^ int(edges[ea, 1]), int(edges[eb, 0]) ^ int(edges[eb, 1])} == {2, 4}]
    reference = project(vertices, distance=m.projection_distance)
    jl, jc = jacobians(reference, edges, corners)
    selected = independent_angles(jl, jc)
    rows = []
    for name in ("prototype_full", "prototype_candidate"):
        for angle in study.scans[name]["angles_deg"]:
            p = project(rotate(vertices, angle), distance=m.projection_distance)
            jl, jc = jacobians(p, edges, corners)
            equal = np.array([jl[i]-jl[ids[0]] for ids in groups.values() for i in ids[1:]])
            q = square_coordinates(angle, m.projection_distance)
            jq = numerical_jacobian(lambda x: square_constraints(x, m.projection_distance)[0], q)
            scheduled = np.vstack((jl, jc[selected]))
            tangent = ((project(rotate(vertices, angle+1e-4), distance=m.projection_distance) -
                        project(rotate(vertices, angle-1e-4), distance=m.projection_distance)) / np.deg2rad(2e-4)).ravel()
            augmented = np.column_stack((scheduled, -scheduled @ tangent))
            row = {"scan": name, "angle_deg": float(angle), "locked_lengths_rank": rank(jl),
                   "equal_family_rank": rank(equal), "equal_family_and_yz_right_angles_rank": rank(np.vstack((equal, jc[yz_corners]))),
                   "scheduled_lengths_and_selected_angles_rank": rank(augmented),
                   "four_square_constraint_rank": rank(jq),
                   "four_square_branch_min_dot": float(square_constraints(q, m.projection_distance)[1].min())}
            row["locked_lengths_internal_infinitesimal_dof"] = 42-row["locked_lengths_rank"]
            row["equal_family_internal_infinitesimal_dof"] = 42-row["equal_family_rank"]
            row["equal_family_and_yz_internal_infinitesimal_dof"] = 42-row["equal_family_and_yz_right_angles_rank"]
            row["scheduled_internal_infinitesimal_dof"] = 43-row["scheduled_lengths_and_selected_angles_rank"]
            row["four_square_infinitesimal_dof"] = 8-row["four_square_constraint_rank"]
            rows.append(row)
    scan = study.scans["prototype_candidate"]
    profiles = []
    eps = 1e-4
    for angle in scan["angles_deg"]:
        for label, scale in (("prototype", scan["summary"]["scale_mm_per_unit"]),
                             ("target", study.scans["target_candidate"]["summary"]["scale_mm_per_unit"])):
            q = square_coordinates(angle, m.projection_distance, scale)
            derivative = (square_coordinates(angle+eps, m.projection_distance, scale) -
                          square_coordinates(angle-eps, m.projection_distance, scale)) / np.deg2rad(2*eps)
            for i, ((x, w), (center, half_width), (dx, db)) in enumerate(zip(SIGNS, q, derivative)):
                profiles.append({"size": label, "frame_id": i, "source_x": x, "source_w": w,
                                 "angle_deg": float(angle), "center_x_mm": center, "half_width_mm": half_width,
                                 "dcenter_dtheta_mm_per_rad": dx, "dhalfwidth_dtheta_mm_per_rad": db})
    inputs = []
    for name, ids in groups.items():
        representative = ids[0]
        angle = scan["angles_deg"]
        slope = []
        for a in angle:
            pair = []
            for t in (a-eps, a+eps):
                p = project(rotate(vertices, t), distance=m.projection_distance)*scan["summary"]["scale_mm_per_unit"]
                va, vb = edges[representative]
                pair.append(np.linalg.norm(p[vb]-p[va]))
            slope.append((pair[1]-pair[0])/np.deg2rad(2*eps))
        slope = np.array(slope)
        reference_slope = float(slope[int(np.argmin(abs(angle)))])
        inputs.append({"name": name, "edge_ids": ids, "representative_edge": representative,
                       "slope_nearest_zero_mm_per_rad": reference_slope,
                       "min_abs_slope_mm_per_rad": float(abs(slope).min()),
                       "slope_min_mm_per_rad": float(slope.min()), "slope_max_mm_per_rad": float(slope.max()),
                       "strictly_monotone_in_sampled_candidate": bool(np.all(slope > 1e-6) or np.all(slope < -1e-6))})
    return {"config_id": study.config.digest(), "rank_tolerance": RANK_TOL,
            "length_coordinate_units_for_rank": "normalized projected coordinates; nonzero Jacobian rows normalized",
            "intrinsic_rotation_shape_dof_at_reference": intrinsic_rotation_dof(vertices, m.projection_distance),
            "selected_local_angle_constraints": [{"corner_index": i, "vertex_id": int(corners[i, 0]),
                "edge_a": int(corners[i, 1]), "edge_b": int(corners[i, 2])} for i in selected],
            "length_families": inputs, "range_counterexample": range_counterexample(study),
            "mobility_samples": rows, "cam_profiles": profiles,
            "limitations": ["Infinitesimal point-model ranks are not finite mobility or stiffness certificates.",
                            "Scheduled lengths/angles must be coupled to one common phase, not independently limited.",
                            "Four-square construction assumes supported, aligned, square frames and positive phase-branch dot products.",
                            "Cam profiles omit follower geometry, pressure angle, backlash, thickness and collision checks.",
                            "The reduction is for xw rotation only; generic 4D rotations do not preserve these guides."]}


def export_constraints(study, output: Path):
    result = constraint_study(study)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in result.items() if key not in ("mobility_samples", "cam_profiles")}
    metadata["config"] = study.config.data()
    metadata["tables"] = {"mobility": "mobility.csv", "cam_profiles": "cam_profiles.csv"}
    metadata["mobility_summary"] = {}
    for name in ("prototype_full", "prototype_candidate"):
        rows = [r for r in result["mobility_samples"] if r["scan"] == name]
        metadata["mobility_summary"][name] = {"samples": len(rows), **{
            key: {"min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
            for key in rows[0] if key.endswith("rank") or key.endswith("dof")}}
    write_json(output / "constraints.json", metadata)
    for name, rows in (("mobility", result["mobility_samples"]), ("cam_profiles", result["cam_profiles"])):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_constraints(study, result, output / "constraints.png")
    plot_frames(study, output / "frame-layout.png")
    candidate = [r for r in result["mobility_samples"] if r["scan"] == "prototype_candidate"]
    keys = ("locked_lengths_internal_infinitesimal_dof", "equal_family_internal_infinitesimal_dof",
            "equal_family_and_yz_internal_infinitesimal_dof", "scheduled_internal_infinitesimal_dof",
            "four_square_infinitesimal_dof")
    labels = ("32 lengths locked", "8 equal-length families", "Families + y/z right angles",
              "Lengths + selected angles coupled to phase", "4 aligned squares: conics + phase")
    lines = ["# Rotation constraint study", "", f"Config `{study.config.digest()}`. xw perspective rotation.", "",
             "Counts below are sampled INFINITESIMAL dimensions, excluding global rigid motion for the point model.", "",
             "| Constraint model | Min | Max |", "| --- | ---: | ---: |"]
    for key, label in zip(keys, labels):
        values = [r[key] for r in candidate]
        lines.append(f"| {label} | {min(values)} | {max(values)} |")
    lines += ["", "## Coupled geometry", "", "Four aligned, expanding square frames have eight coordinates (X, B per frame).",
              "Four conic constraints plus three source-phase equalities leave one local motion.",
              "Positive phase dot products select the intended assembly branch.",
              f"The selected local complement contains {len(result['selected_local_angle_constraints'])} corner-angle constraints.",
              "They are scalar target functions, NOT a count of physical hinges or actuators.", "",
              "## Limits are insufficient", "", str(result["range_counterexample"]), "",
              "## Limits of this result", ""] + [f"- {s}" for s in result["limitations"]]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def plot_frames(study, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = study.scans["prototype_candidate"]["summary"]["scale_mm_per_unit"]
    fig = plt.figure(figsize=(13, 5), layout="constrained")
    colors = ("#167d9a", "#c87832", "#6b62ac", "#378c64")
    for column, angle in enumerate((0, study.config.model.candidate_end_deg)):
        ax = fig.add_subplot(1, 2, column+1, projection="3d")
        q = square_coordinates(angle, study.config.model.projection_distance, scale)
        points = square_vertices(q)
        for a, b in study.edges:
            axis = int(a) ^ int(b)
            frame = SIGNS.index((int(study.vertices[a, 0]), int(study.vertices[a, 3])))
            ax.plot(*points[[a, b]].T, color=colors[frame] if axis in (2, 4) else "#a3b1b8",
                    linewidth=2.5 if axis in (2, 4) else .8)
        for i, ((x, w), (center, half)) in enumerate(zip(SIGNS, q)):
            ax.text(center, half, half, f" F{i}", color=colors[i])
        ax.set(title=f"4 aligned square frames / {angle:g} deg", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
        ax.set_box_aspect((1, 1, 1))
        radius = study.config.model.size_mm / 2
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-radius, radius)
        ax.view_init(elev=20, azim=-55)
    fig.suptitle("Each colored square: one X position + one half-width B / gray links follow")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_constraints(study, result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = study.config.model
    scale = study.scans["prototype_candidate"]["summary"]["scale_mm_per_unit"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    full = np.array([square_coordinates(a, m.projection_distance, scale) for a in np.linspace(0, 360, 361)])
    axs[0, 0].plot(full[:, 0, 0], full[:, 0, 1], color="#91a5b2", label="common conic")
    q = square_coordinates(0, m.projection_distance, scale)
    for i, (x, w) in enumerate(SIGNS):
        axs[0, 0].scatter(*q[i], label=f"frame {i}: x{x:+d}, w{w:+d}")
    axs[0, 0].set(title="Four square frames on one conic (reference)", xlabel="Axial center X (mm)", ylabel="Half-width B (mm)")
    axs[0, 0].legend(fontsize=8)
    for i in range(4):
        p = [r for r in result["cam_profiles"] if r["size"] == "prototype" and r["frame_id"] == i]
        axs[0, 1].plot([r["angle_deg"] for r in p], [r["center_x_mm"] for r in p], label=f"X{i}")
        axs[0, 1].plot([r["angle_deg"] for r in p], [r["half_width_mm"] for r in p], linestyle="--", label=f"B{i}")
    axs[0, 1].set(title="Eight follower targets, one common input", xlabel="4D phase (deg)", ylabel="Position (mm)")
    axs[0, 1].legend(ncol=4, fontsize=8)
    for name, ids in edge_groups(study.vertices, study.edges).items():
        lengths = study.scans["prototype_candidate"]["lengths_mm"][:, ids[0]]
        theta = np.deg2rad(study.scans["prototype_candidate"]["angles_deg"])
        axs[1, 0].plot(np.rad2deg(theta), np.gradient(lengths, theta), label=f"E{ids[0]}")
    axs[1, 0].axhline(0, color="#666", linewidth=.7)
    axs[1, 0].set(title="Single-length input: watch zero slopes / reversals", xlabel="4D phase (deg)", ylabel="dL/dtheta (mm/rad)")
    axs[1, 0].legend(ncol=4, fontsize=8)
    rows = [r for r in result["mobility_samples"] if r["scan"] == "prototype_candidate"]
    for key, label in (("equal_family_internal_infinitesimal_dof", "8 equal-length families"),
                       ("equal_family_and_yz_internal_infinitesimal_dof", "+ yz right angles"),
                       ("locked_lengths_internal_infinitesimal_dof", "32 lengths locked"),
                       ("scheduled_internal_infinitesimal_dof", "coupled phase + selected angles")):
        axs[1, 1].plot([r["angle_deg"] for r in rows], [r[key] for r in rows], label=label)
    axs[1, 1].set(title="Sampled infinitesimal motion / not finite mobility", xlabel="4D phase (deg)", ylabel="Internal infinitesimal dimension")
    axs[1, 1].legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"Rotation constraints / {study.config.digest()} / point geometry only")
    fig.savefig(path, dpi=140)
    plt.close(fig)
