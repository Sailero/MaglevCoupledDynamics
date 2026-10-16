"""High-level simulate / resume: uniform output, retune segments, observations."""

from __future__ import annotations

import copy

import numpy as np

from maglev_sim.plant.config import Config, set_path
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.integrate import COMPLETED, integrate_segment
from maglev_sim.plant.model import observe
from maglev_sim.plant.state import layout_from_config
from maglev_sim.plant.trajectory import Trajectory


def _observe_series(
    times: list[float],
    states: list[np.ndarray],
    configs: list[Config],
    excitation: Excitation,
) -> dict[str, np.ndarray]:
    n = len(times)
    layout = layout_from_config(configs[0])
    n_modes = layout.n_modes
    out = {
        "z_car": np.empty(n),
        "z_magnet": np.empty(n),
        "v_car": np.empty(n),
        "v_magnet": np.empty(n),
        "a_car": np.empty(n),
        "a_magnet": np.empty(n),
        "gap_raw": np.empty(n),
        "beam_displacement_at_magnet": np.empty(n),
        "beam_velocity_at_magnet": np.empty(n),
        "current": np.empty(n),
        "current_command": np.empty(n),
        "current_raw": np.empty(n),
        "magnetic_force": np.empty(n),
        "q": np.empty((n, n_modes)),
        "p": np.empty((n, n_modes)),
        "irregularity": np.empty(n),
        "force_x": np.empty(n),
        "force_P": np.empty(n),
        "xi": np.empty(n),
        "speed": np.empty(n),
        "saturated": np.empty(n, dtype=np.int8),
    }
    for k, (t, y, cfg) in enumerate(zip(times, states, configs)):
        obs = observe(float(t), y, cfg, excitation)
        out["z_car"][k] = obs.z_car
        out["z_magnet"][k] = obs.z_magnet
        out["v_car"][k] = obs.v_car
        out["v_magnet"][k] = obs.v_magnet
        out["a_car"][k] = obs.a_car
        out["a_magnet"][k] = obs.a_magnet
        out["gap_raw"][k] = obs.gap_raw
        out["beam_displacement_at_magnet"][k] = obs.beam_displacement_at_magnet
        out["beam_velocity_at_magnet"][k] = obs.beam_velocity_at_magnet
        out["current"][k] = obs.current
        out["current_command"][k] = obs.current_command
        out["current_raw"][k] = obs.current_raw
        out["magnetic_force"][k] = obs.magnetic_force
        out["q"][k] = obs.q
        out["p"][k] = obs.p
        out["irregularity"][k] = obs.irregularity
        out["force_x"][k] = obs.force_x
        out["force_P"][k] = obs.force_P
        out["xi"][k] = obs.xi
        out["speed"][k] = obs.speed
        out["saturated"][k] = 1 if obs.saturated else 0
    return out


def _segment_plan(config: Config, t0: float, t1: float) -> list[tuple[float, float, Config]]:
    events = sorted((e for e in config.retune if t0 < e.t < t1), key=lambda e: e.t)
    bounds = [t0] + [e.t for e in events] + [t1]
    cfgs = [config]
    cfg = config
    for event in events:
        cfg = set_path(cfg, event.path, event.value)
        cfg.validate()
        cfgs.append(cfg)
    return [(bounds[i], bounds[i + 1], cfgs[i]) for i in range(len(cfgs))]


def simulate(
    config: Config,
    initial_state: np.ndarray,
    excitation: Excitation,
    output_times: np.ndarray,
) -> Trajectory:
    config.validate()
    times = np.asarray(output_times, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("output_times must be a 1-D array")
    if times.size > 1 and np.any(np.diff(times) < 0.0):
        raise ValueError("output_times must be nondecreasing")

    layout = layout_from_config(config)
    y = np.asarray(initial_state, dtype=float).reshape(-1)
    if y.size != layout.dim:
        raise ValueError(f"initial_state dim {y.size} != {layout.dim}")

    t0 = float(times[0])
    t1 = float(times[-1])
    excitation.validate_interval(t0, t1)
    plan = _segment_plan(config, t0, t1)
    retune_log = [
        {"t": ev.t, "path": ev.path, "value": ev.value}
        for ev in sorted(config.retune, key=lambda e: e.t)
        if t0 < ev.t < t1
    ]

    t_hist = [t0]
    y_hist = [y.copy()]
    cfg_hist = [plan[0][2]]
    reason = COMPLETED
    message = "ok"

    for ta, tb, cfg_seg in plan:
        mask = (times > ta + 1e-15) & (times <= tb + 1e-15)
        interior = times[mask]
        t_eval = np.unique(np.concatenate(([ta], interior, [tb])))
        if t_eval.size == 1:
            continue
        seg = integrate_segment(cfg_seg, y_hist[-1], t_eval, excitation)
        start = 1 if seg.t.size and abs(seg.t[0] - t_hist[-1]) < 1e-12 else 0
        for tt, yy in zip(seg.t[start:], seg.y[start:]):
            t_hist.append(float(tt))
            y_hist.append(np.asarray(yy, dtype=float).reshape(-1))
            cfg_hist.append(cfg_seg)
        reason = seg.reason
        message = seg.message
        if reason != COMPLETED:
            break

    arrays = _observe_series(t_hist, y_hist, cfg_hist, excitation)
    t_arr = np.asarray(t_hist, dtype=float)
    y_arr = np.vstack(y_hist)
    dt = float(times[1] - times[0]) if times.size > 1 else 0.0
    reached_end = t_arr[-1] >= t1 - 0.51 * dt
    completed = bool(reason == COMPLETED and bool(reached_end))
    term = COMPLETED if completed else reason

    return Trajectory(
        t=t_arr,
        state=y_arr,
        state_names=layout.names,
        completed=completed,
        t_end_actual=float(t_arr[-1]),
        termination_reason=term,
        integrator_message=message,
        config=copy.deepcopy(config),
        excitation_meta=excitation.to_meta(),
        retune_log=retune_log,
        **arrays,
    )


def resume(
    config: Config,
    state: np.ndarray,
    excitation: Excitation,
    output_times: np.ndarray,
) -> Trajectory:
    """从已保存状态继续积分，接口与 simulate 相同。"""
    return simulate(config, state, excitation, output_times)
