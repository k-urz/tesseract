"""Small, validated TOML configuration shared by every output."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Model:
    plane: str = "xw"
    projection: str = "perspective"
    projection_distance: float = 4.0
    size_mm: float = 300.0
    target_size_mm: float = 150.0
    coarse_step_deg: float = 1.0
    candidate_start_deg: float = -15.0
    candidate_end_deg: float = 15.0
    fine_step_deg: float = 0.1


@dataclass(frozen=True)
class Hardware:
    hub_radius_mm: float = 18.0
    tube_width_mm: float = 10.0
    tube_wall_mm: float = 1.5
    slide_clearance_mm: float = 0.3
    min_overlap_mm: float = 20.0
    end_wall_mm: float = 2.0
    ball_radius_mm: float = 5.0
    neck_radius_mm: float = 2.0
    socket_clearance_mm: float = 0.2
    bolt_diameter_mm: float = 3.0
    hole_clearance_mm: float = 0.3
    nut_across_flats_mm: float = 5.5
    nut_thickness_mm: float = 2.4
    fit_clearances_mm: tuple[float, ...] = (0.15, 0.25, 0.35, 0.5)

    @property
    def rod_radius_mm(self):
        # Circumscribed circle, not the inscribed circle of the square tube.
        return self.tube_width_mm / math.sqrt(2)

    @property
    def inner_width_mm(self):
        return self.tube_width_mm - 2 * self.tube_wall_mm - 2 * self.slide_clearance_mm


@dataclass(frozen=True)
class Config:
    model: Model = Model()
    hardware: Hardware = Hardware()

    def validate(self):
        m, h = self.model, self.hardware
        if m.plane not in ("xy", "xz", "xw", "yz", "yw", "zw"):
            raise ValueError("plane must be one of xy,xz,xw,yz,yw,zw")
        if m.projection not in ("perspective", "orthographic"):
            raise ValueError("projection must be perspective or orthographic")
        for obj in (m, h):
            for f in fields(obj):
                value = getattr(obj, f.name)
                if isinstance(obj, Model) and f.name in ("plane", "projection"):
                    continue
                values = value if isinstance(value, (list, tuple)) else (value,)
                if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                    raise ValueError(f"{f.name} must contain finite numbers")
        for name in ("projection_distance", "size_mm", "target_size_mm", "coarse_step_deg", "fine_step_deg"):
            if getattr(m, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if m.fine_step_deg >= m.coarse_step_deg:
            raise ValueError("fine_step_deg must be smaller than coarse_step_deg")
        if not -360 <= m.candidate_start_deg < m.candidate_end_deg <= 360:
            raise ValueError("candidate must be an increasing interval within [-360,360]")
        if 360 / m.coarse_step_deg > 10000 or (m.candidate_end_deg - m.candidate_start_deg) / m.fine_step_deg > 10000:
            raise ValueError("limit each scan to 10000 intervals")
        for f in fields(h):
            value = getattr(h, f.name)
            values = value if isinstance(value, (list, tuple)) else (value,)
            if any(v <= 0 for v in values):
                raise ValueError(f"{f.name} must be positive")
        if h.inner_width_mm <= h.bolt_diameter_mm + h.hole_clearance_mm:
            raise ValueError("tube bore/inner rod must leave material around the stop-pin hole")
        if h.neck_radius_mm >= h.ball_radius_mm:
            raise ValueError("neck must be narrower than the ball")
        if min(m.size_mm, m.target_size_mm) <= 2 * h.hub_radius_mm:
            raise ValueError("model envelope must exceed the hub diameter")
        if h.ball_radius_mm + h.socket_clearance_mm + 3 >= h.hub_radius_mm:
            raise ValueError("hub envelope is too small for the trial ball housing")
        return self

    def data(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.data(), sort_keys=True).encode()).hexdigest()[:16]


def load_config(path: Path | None = None) -> Config:
    if path is None:
        return Config().validate()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    unknown = data.keys() - {"model", "hardware"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    objects = []
    for section, cls in (("model", Model), ("hardware", Hardware)):
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a table")
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown {section} settings: {sorted(unknown)}")
        if "fit_clearances_mm" in values:
            values["fit_clearances_mm"] = tuple(values["fit_clearances_mm"])
        objects.append(cls(**values))
    return Config(*objects).validate()
