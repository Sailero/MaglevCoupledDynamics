"""solve_ivp wrapper with termination events. No observation stacking here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from maglev_sim.plant.config import Config
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.model import observe, rhs
from maglev_sim.plant.state import layout_from_config

GAP_MIN = "gap_min"
GAP_MAX = "gap_max"
LEFT_BEAM = "left_beam"
NON_FINITE = "non_finite"
INTEGRATOR_FAILED = "integrator_failed"
COMPLETED = "completed"


@dataclass
class SegmentResult:
    t: NDArray[np.float64]
    y: NDArray[np.float64]
    reason: str
    message: str
    success: bool


def _events(config: Config, excitation: Excitation):
    layout = layout_from_config(config)
    gmin = config.magnet.gap_min
    gmax = config.magnet.gap_max
    L = config.beam.L
    moving = config.motion.mode == "moving"

    def gap_min(t, y):
        return observe(t, y, config, excitation).gap_raw - gmin

    gap_min.terminal = True
    gap_min.direction = -1.0

    def gap_max(t, y):
        return gmax - observe(t, y, config, excitation).gap_raw

    gap_max.terminal = True
    gap_max.direction = -1.0

    def left_low(t, y):
        del y
        return excitation.position(t)

    left_low.terminal = True
    left_low.direction = -1.0

    def left_high(t, y):
        del y
        return L - excitation.position(t)

    left_high.terminal = True
    left_high.direction = -1.0

    def finite_state(t, y):
        del t
        return 1.0 if np.all(np.isfinite(y)) else -1.0

    finite_state.terminal = True
    finite_state.direction = -1.0

    events = [gap_min, gap_max, finite_state]
    if moving:
        events.extend([left_low, left_high])
    names = [GAP_MIN, GAP_MAX, NON_FINITE]
    if moving:
        names.extend([LEFT_BEAM, LEFT_BEAM])
    return events, names, layout


def integrate_segment(
    config: Config,
    y0: np.ndarray,
    t_eval: np.ndarray,
    excitation: Excitation,
) -> SegmentResult:
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.size == 0:
        raise ValueError("t_eval is empty")
    if t_eval.size == 1:
        y = np.asarray(y0, dtype=float).reshape(1, -1)
        return SegmentResult(
            t=t_eval.copy(),
            y=y,
            reason=COMPLETED,
            message="single sample",
            success=True,
        )

    events, names, _layout = _events(config, excitation)
    method = config.integrator.method
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, config, excitation),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(y0, dtype=float).reshape(-1),
        method=method,
        t_eval=t_eval,
        events=events,
        rtol=config.integrator.rtol,
        atol=config.integrator.atol,
        max_step=config.integrator.max_step,
        dense_output=True,
    )

    reason = COMPLETED
    if sol.t_events:
        for idx, arr in enumerate(sol.t_events):
            if arr.size:
                reason = names[idx]
                break
    if not sol.success and reason == COMPLETED:
        reason = INTEGRATOR_FAILED

    t = np.asarray(sol.t, dtype=float)
    y = np.asarray(sol.y, dtype=float).T
    if sol.status == 1 and sol.t_events:
        t_hit = min(float(arr[-1]) for arr in sol.t_events if arr.size)
        if t.size == 0 or t_hit > t[-1] + 1e-15:
            if sol.sol is not None:
                y_hit = sol.sol(t_hit)
                t = np.append(t, t_hit)
                y = np.vstack([y, y_hit])
        elif t.size and abs(t[-1] - t_hit) > 1e-12 and sol.sol is not None:
            y[-1] = sol.sol(t_hit)
            t[-1] = t_hit

    if y.size and not np.all(np.isfinite(y[-1])):
        reason = NON_FINITE

    return SegmentResult(
        t=t,
        y=y,
        reason=reason,
        message=str(sol.message),
        success=bool(sol.success) or reason != INTEGRATOR_FAILED,
    )
