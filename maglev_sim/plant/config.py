"""Nested simulation configuration. All values are SI unless noted."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

MU_0 = 4.0e-7 * math.pi
MODEL_VERSION = "gap_pid_modal_v1"


def _from_dict(cls, data: dict[str, Any]):
    if data is None:
        data = {}
    kwargs: dict[str, Any] = {}
    valid = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(valid) - {"i0", "kappa", "F0"}
    if unknown:
        raise ValueError(f"{cls.__name__} unknown keys: {sorted(unknown)}")
    for name in valid:
        if name not in data:
            continue
        kwargs[name] = data[name]
    return cls(**kwargs)


@dataclass
class MetaConfig:
    model_version: str = MODEL_VERSION
    parameter_source: str = "demo_not_hs_maglev"
    notes: str = (
        "演示参数，仅用于接口与数值检查。"
        "不是已经验证的高速磁浮稳定性算例。"
    )
    coordinate: str = "downward_positive"
    units: str = "SI"
    name: str = "demo"


@dataclass
class VehicleConfig:
    m_c: float = 500.0
    m_m: float = 250.0
    k_s: float = 1.2e5
    c_s: float = 3.1e3
    g: float = 9.81

    def __post_init__(self) -> None:
        self.m_c = float(self.m_c)
        self.m_m = float(self.m_m)
        self.k_s = float(self.k_s)
        self.c_s = float(self.c_s)
        self.g = float(self.g)

    @property
    def F0(self) -> float:
        return (self.m_c + self.m_m) * self.g


@dataclass
class BeamConfig:
    L: float = 24.0
    E: float = 2.1e11
    I_b: float = 0.008
    mu_b: float = 400.0
    zeta: float | list[float] = 0.02
    n_modes: int = 5
    enabled: bool = True

    def __post_init__(self) -> None:
        self.L = float(self.L)
        self.E = float(self.E)
        self.I_b = float(self.I_b)
        self.mu_b = float(self.mu_b)
        self.n_modes = int(self.n_modes)
        self.enabled = bool(self.enabled)
        if isinstance(self.zeta, list):
            self.zeta = [float(z) for z in self.zeta]
        else:
            self.zeta = float(self.zeta)

    def zeta_j(self, j: int) -> float:
        """Modal damping ratio for 1-based mode index j."""
        if isinstance(self.zeta, (int, float)):
            return float(self.zeta)
        if j < 1 or j > len(self.zeta):
            raise IndexError(f"no damping ratio for mode {j}")
        return float(self.zeta[j - 1])


@dataclass
class MagnetConfig:
    n_w: float = 300.0
    A_p: float = 0.02
    delta_0: float = 0.008
    I_min: float = 0.5
    I_max: float = 60.0
    gap_min: float = 0.001
    gap_max: float = 0.025
    mu_0: float = MU_0
    delta_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        self.n_w = float(self.n_w)
        self.A_p = float(self.A_p)
        self.delta_0 = float(self.delta_0)
        self.I_min = float(self.I_min)
        self.I_max = float(self.I_max)
        self.gap_min = float(self.gap_min)
        self.gap_max = float(self.gap_max)
        self.mu_0 = float(self.mu_0)
        self.delta_floor = float(self.delta_floor)

    @property
    def kappa(self) -> float:
        return self.mu_0 * self.n_w**2 * self.A_p / 4.0


@dataclass
class ControllerConfig:
    name: str = "gap_pid"
    Kp: float = 6000.0
    Ki: float = 2000.0
    Kd: float = 40.0
    tau_d: float = 0.005
    tau_i: float = 0.003

    def __post_init__(self) -> None:
        self.Kp = float(self.Kp)
        self.Ki = float(self.Ki)
        self.Kd = float(self.Kd)
        self.tau_d = float(self.tau_d)
        self.tau_i = float(self.tau_i)


@dataclass
class MotionConfig:
    mode: str = "fixed"
    xi: float = 12.0
    v: float = 0.0

    def __post_init__(self) -> None:
        self.xi = float(self.xi)
        self.v = float(self.v)


@dataclass
class IrregularityConfig:
    type: str = "constant"
    value: float = 0.0
    amplitude: float = 0.0
    wavelength: float = 8.0
    phase: float = 0.0
    x: list[float] | None = None
    r: list[float] | None = None

    def __post_init__(self) -> None:
        self.value = float(self.value)
        self.amplitude = float(self.amplitude)
        self.wavelength = float(self.wavelength)
        self.phase = float(self.phase)


@dataclass
class ForceConfig:
    type: str = "none"
    x_f: float = 12.0
    amplitude: float = 0.0
    frequency_hz: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    t: list[float] | None = None
    P: list[float] | None = None
    seed: int | None = None
    dt: float = 0.001

    def __post_init__(self) -> None:
        self.x_f = float(self.x_f)
        self.amplitude = float(self.amplitude)
        self.frequency_hz = float(self.frequency_hz)
        self.t_start = float(self.t_start)
        self.t_end = float(self.t_end)
        self.dt = float(self.dt)
        if self.seed is not None:
            self.seed = int(self.seed)


@dataclass
class InitialConfig:
    mode: str = "equilibrium"
    delta: dict[str, float] = field(default_factory=dict)
    random_std: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.delta = {k: float(v) for k, v in dict(self.delta or {}).items()}
        self.random_std = {k: float(v) for k, v in dict(self.random_std or {}).items()}


@dataclass
class RetuneEvent:
    t: float
    path: str
    value: float

    def __post_init__(self) -> None:
        self.t = float(self.t)
        self.value = float(self.value)


@dataclass
class IntegratorConfig:
    method: str = "RK45"
    rtol: float = 1.0e-8
    atol: float = 1.0e-10
    max_step: float = 1.0e-3

    def __post_init__(self) -> None:
        self.rtol = float(self.rtol)
        self.atol = float(self.atol)
        self.max_step = float(self.max_step)


@dataclass
class OutputConfig:
    dt: float = 1.0e-3
    t_end: float = 5.0
    t_start: float = 0.0

    def __post_init__(self) -> None:
        self.dt = float(self.dt)
        self.t_end = float(self.t_end)
        self.t_start = float(self.t_start)


@dataclass
class SplitConfig:
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 0


@dataclass
class Config:
    meta: MetaConfig = field(default_factory=MetaConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    magnet: MagnetConfig = field(default_factory=MagnetConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    irregularity: IrregularityConfig = field(default_factory=IrregularityConfig)
    force: ForceConfig = field(default_factory=ForceConfig)
    initial: InitialConfig = field(default_factory=InitialConfig)
    retune: list[RetuneEvent] = field(default_factory=list)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    @property
    def i0(self) -> float:
        kappa = self.magnet.kappa
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        return self.magnet.delta_0 * math.sqrt(self.vehicle.F0 / kappa)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        data = dict(data or {})
        for derived in ("i0", "kappa", "F0"):
            data.pop(derived, None)
        retune_raw = data.pop("retune", []) or []
        nested_types = {
            "meta": MetaConfig,
            "vehicle": VehicleConfig,
            "beam": BeamConfig,
            "magnet": MagnetConfig,
            "controller": ControllerConfig,
            "motion": MotionConfig,
            "irregularity": IrregularityConfig,
            "force": ForceConfig,
            "initial": InitialConfig,
            "integrator": IntegratorConfig,
            "output": OutputConfig,
            "split": SplitConfig,
        }
        extra = set(data) - set(nested_types) - {"retune"}
        if extra:
            raise ValueError(f"Config unknown keys: {sorted(extra)}")
        nested = {
            name: _from_dict(typ, data[name])
            for name, typ in nested_types.items()
            if name in data
        }
        cfg = cls(**nested)
        cfg.retune = [
            item if isinstance(item, RetuneEvent) else RetuneEvent(**item)
            for item in retune_raw
        ]
        return cfg

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["i0"] = self.i0
        payload["kappa"] = self.magnet.kappa
        payload["F0"] = self.vehicle.F0
        return payload

    def validate(self) -> None:
        if self.vehicle.m_c <= 0.0 or self.vehicle.m_m <= 0.0:
            raise ValueError("masses must be positive")
        if self.beam.n_modes < 0:
            raise ValueError("n_modes must be >= 0")
        if self.beam.L <= 0.0:
            raise ValueError("beam length must be positive")
        if not 0.0 < self.motion.xi < self.beam.L:
            raise ValueError("xi must lie in (0, L)")
        if self.motion.mode not in {"fixed", "moving"}:
            raise ValueError("motion.mode must be 'fixed' or 'moving'")
        if self.controller.tau_d <= 0.0 or self.controller.tau_i <= 0.0:
            raise ValueError("tau_d and tau_i must be positive")
        if self.magnet.gap_min <= 0.0 or self.magnet.gap_max <= self.magnet.gap_min:
            raise ValueError("invalid gap limits")
        if self.magnet.delta_floor <= 0.0:
            raise ValueError("delta_floor must be positive")
        i0 = self.i0
        if not (self.magnet.I_min <= i0 <= self.magnet.I_max):
            raise ValueError(
                f"equilibrium current i0={i0:.6g} A outside "
                f"[{self.magnet.I_min}, {self.magnet.I_max}]"
            )
        if self.magnet.delta_0 <= 0.0:
            raise ValueError("delta_0 must be positive")

    def output_times(self) -> Any:
        import numpy as np

        t0 = self.output.t_start
        t1 = self.output.t_end
        dt = self.output.dt
        if dt <= 0.0 or t1 < t0:
            raise ValueError("invalid output times")
        n = int(round((t1 - t0) / dt))
        return t0 + dt * np.arange(n + 1)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config.from_dict(data)
    if not cfg.meta.name or cfg.meta.name == "demo":
        cfg.meta.name = path.stem
    cfg.validate()
    return cfg


def demo_config() -> Config:
    cfg = Config()
    cfg.validate()
    return cfg


def set_path(config: Config, path: str, value: Any) -> Config:
    """Return a copy with a dotted attribute path updated."""
    import copy

    cfg = copy.deepcopy(config)
    parts = path.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    return cfg
