"""External inputs U(t): irregularity, point force, and prescribed xi(t)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from maglev_sim.plant.config import Config, ForceConfig, IrregularityConfig, MotionConfig


def _as_interp(t: NDArray[np.float64], y: NDArray[np.float64], *, strict=False):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 2:
        raise ValueError("precomputed force series must have length >= 2")
    if not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError("interpolation arrays must be finite")
    if strict and np.any(np.diff(t) <= 0):
        raise ValueError("time knots must be strictly increasing")
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    def interp(query: float) -> float:
        return float(np.interp(query, t, y))

    return interp


@dataclass
class Excitation:
    """Deterministic excitation. Constructed before integration; no RNG in rhs."""

    motion: MotionConfig
    irregularity: IrregularityConfig
    force: ForceConfig
    beam_length: float
    _force_fn: object = field(repr=False, default=None)
    _r_fn: object = field(repr=False, default=None)
    _dr_fn: object = field(repr=False, default=None)

    @classmethod
    def from_config(cls, config: Config, seed: int | None = None) -> Excitation:
        for spec, kind in ((config.irregularity, 'time_series'), (config.force, 'precomputed')):
            if spec.type == kind and spec.t is not None:
                times = np.asarray(spec.t, dtype=float)
                if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
                    raise ValueError('time knots must be finite 1-D arrays of length >= 2')
                if np.any(np.diff(times) <= 0):
                    raise ValueError('time knots must be strictly increasing')
                if times[0] > config.output.t_start or times[-1] < config.output.t_end:
                    raise ValueError('time series must cover the integration interval')
        force_fn = _build_force(config.force, config.output.t_end, seed)
        r_fn, dr_fn = _build_irregularity(config.irregularity)
        return cls(
            motion=config.motion,
            irregularity=config.irregularity,
            force=config.force,
            beam_length=config.beam.L,
            _force_fn=force_fn,
            _r_fn=r_fn,
            _dr_fn=dr_fn,
        )

    def position(self, t: float) -> float:
        if self.motion.mode == "fixed":
            return float(self.motion.xi)
        return float(self.motion.xi + self.motion.v * t)

    def speed(self, t: float) -> float:
        del t
        if self.motion.mode == "fixed":
            return 0.0
        return float(self.motion.v)

    def irregularity_value(self, xi: float, t: float | None = None) -> float:
        if self.irregularity.type == 'time_series':
            if t is None:
                raise ValueError('time_series irregularity requires time t')
            return float(self._r_fn(t))
        return float(self._r_fn(xi))  # type: ignore[misc]

    def validate_interval(self, t0: float, t1: float) -> None:
        for spec, kind in ((self.irregularity, 'time_series'), (self.force, 'precomputed')):
            if spec.type == kind and spec.t is not None:
                if spec.t[0] > t0 or spec.t[-1] < t1:
                    raise ValueError('time series must cover the integration interval')

    def irregularity_slope(self, xi: float) -> float:
        return float(self._dr_fn(xi))  # type: ignore[misc]

    def point_force(self, t: float) -> tuple[float, float]:
        P = float(self._force_fn(t))  # type: ignore[misc]
        return float(self.force.x_f), P

    def to_meta(self) -> dict:
        return {
            "motion": {
                "mode": self.motion.mode,
                "xi0": self.motion.xi,
                "v": self.motion.v,
            },
            "irregularity": {
                "type": self.irregularity.type,
                "value": self.irregularity.value,
                "amplitude": self.irregularity.amplitude,
                "wavelength": self.irregularity.wavelength,
                "phase": self.irregularity.phase,
            },
            "force": {
                "type": self.force.type,
                "x_f": self.force.x_f,
                "amplitude": self.force.amplitude,
                "frequency_hz": self.force.frequency_hz,
                "t_start": self.force.t_start,
                "t_end": self.force.t_end,
                "seed": self.force.seed,
            },
        }


def _build_force(force: ForceConfig, t_end: float, seed: int | None):
    kind = force.type
    if kind in {"none", None, ""}:
        return lambda t: 0.0
    if kind == "harmonic":
        omega = 2.0 * np.pi * force.frequency_hz
        amp = force.amplitude
        return lambda t, omega=omega, amp=amp: amp * np.sin(omega * t)
    if kind == "pulse":
        amp = force.amplitude
        t0, t1 = force.t_start, force.t_end
        return lambda t, amp=amp, t0=t0, t1=t1: amp if t0 <= t <= t1 else 0.0
    if kind == "precomputed":
        if (force.t is None) != (force.P is None):
            raise ValueError('precomputed force requires both t and P arrays')
        if force.t is None or force.P is None:
            used_seed = force.seed if force.seed is not None else seed
            if used_seed is None:
                raise ValueError("precomputed force needs series or seed")
            dt = force.dt if force.dt > 0.0 else 0.001
            t = np.arange(0.0, t_end + dt, dt)
            rng = np.random.default_rng(used_seed)
            P = force.amplitude * rng.standard_normal(t.size)
            return _as_interp(t, P)
        return _as_interp(np.asarray(force.t), np.asarray(force.P), strict=True)
    raise ValueError(f"unknown force type {kind!r}")


def _build_irregularity(spec: IrregularityConfig):
    kind = spec.type
    if kind == 'time_series':
        if spec.t is None or spec.r is None:
            raise ValueError('time_series irregularity needs t and r')
        return _as_interp(np.asarray(spec.t), np.asarray(spec.r), strict=True), lambda xi: 0.0
    if kind in {"constant", None, ""}:
        value = float(spec.value)
        return (lambda xi, value=value: value, lambda xi: 0.0)
    if kind == "spatial_sine":
        amp = float(spec.amplitude)
        lam = float(spec.wavelength)
        phase = float(spec.phase)
        if lam <= 0.0:
            raise ValueError("irregularity wavelength must be positive")
        k = 2.0 * np.pi / lam

        def r_fn(xi, amp=amp, k=k, phase=phase):
            return amp * np.sin(k * xi + phase)

        def dr_fn(xi, amp=amp, k=k, phase=phase):
            return amp * k * np.cos(k * xi + phase)

        return r_fn, dr_fn
    if kind == "spatial_table":
        if spec.x is None or spec.r is None:
            raise ValueError("spatial_table irregularity needs x and r")
        x = np.asarray(spec.x, dtype=float)
        r = np.asarray(spec.r, dtype=float)
        r_fn = _as_interp(x, r)
        dr = np.gradient(r, x)

        def dr_fn(xi, x=x, dr=dr):
            return float(np.interp(xi, x, dr))

        return r_fn, dr_fn
    raise ValueError(f"unknown irregularity type {kind!r}")
