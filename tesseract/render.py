"""Portable scientific outputs; the HTML viewer needs no server or CDN."""

import json
import hashlib
from pathlib import Path
import numpy as np

from .analysis import clean, write_json


def render(study, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    scan = study.scans["prototype_candidate"]
    target = study.scans["target_candidate"]
    angles = scan["angles_deg"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axs[0, 0].plot(angles, scan["lengths_mm"], alpha=.55, linewidth=.8)
    axs[0, 0].set(title="Each edge: prototype", ylabel="Center-to-center length (mm)")
    axs[0, 1].plot(angles, scan["joint_angles_deg"], alpha=.35, linewidth=.8)
    axs[0, 1].set(title="Pairwise angles at vertices", ylabel="Angle (deg)")
    for label, s in (("Prototype", scan), ("Target", target)):
        axs[1, 0].plot(angles, [c["minimum_clearance_mm"] for c in s["checks"]], label=label)
    axs[1, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axs[1, 0].set(title="Nominal envelope clearance", ylabel="Clearance (mm)")
    axs[1, 0].legend()
    ranges = np.array([[e["min_mm"], e["max_mm"]] for e in scan["edges"]], dtype=float)
    axs[1, 1].vlines(np.arange(32), ranges[:, 0], ranges[:, 1], color="#168e9c", linewidth=3)
    axs[1, 1].scatter(np.arange(32), ranges[:, 0], s=8, color="#168e9c")
    axs[1, 1].set(title="Required range per edge", xlabel="Edge ID", ylabel="Length (mm)")
    for ax in axs.flat:
        ax.grid(alpha=.2)
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.set_xlabel("4D rotation (deg)")
    fig.suptitle(f"Tesseract motion study / config {study.config.digest()}")
    fig.savefig(output / "requirements.png", dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4), layout="constrained")
    for column, angle in enumerate([study.config.model.candidate_start_deg, 0, study.config.model.candidate_end_deg]):
        idx = int(np.argmin(np.abs(angles - angle)))
        points = scan["points_mm"][idx]
        ax = fig.add_subplot(1, 3, column+1, projection="3d")
        for edge_id, (a, b) in enumerate(study.edges):
            color = "#e15e46" if edge_id in scan["checks"][idx]["colliding_edges"] else "#168e9c"
            ax.plot(*points[[a, b]].T, color=color, linewidth=1.7)
        ax.scatter(*points.T, s=15, color="#163f56")
        radius = study.config.model.size_mm/2
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-radius, radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set(title=f"{angles[idx]:g} deg", xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    fig.savefig(output / "motion.png", dpi=140)
    plt.close(fig)

    payload = {"config_id": study.config.digest(), "config": study.config.data(),
               "edges": study.edges, "scans": {}}
    for name in ("prototype_full", "prototype_candidate", "target_full", "target_candidate"):
        s = study.scans[name]
        payload["scans"][name] = {"summary": s["summary"],
            "frames": [{"angle": a, "points": p, "lengths": lengths, "check": check}
                       for a, p, lengths, check in zip(s["angles_deg"], s["points_mm"], s["lengths_mm"], s["checks"])]}
    data = json.dumps(clean(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    (output / "viewer.html").write_text(template.replace("/*__STUDY__*/null", data), encoding="utf-8")
    # A later analyze/cad run may use another configuration in the same folder.
    # Bind every rendered artifact to its configuration and exact bytes so the
    # downloadable manufacturing pack cannot silently mix different studies.
    write_json(output / "render_manifest.json", {
        "config_id": study.config.digest(),
        "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                   for name in ("viewer.html", "requirements.png", "motion.png")},
    })
