"""Pure equilibrium, rhs, and observation. No hidden integrator memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from maglev_sim.plant.config import Config
from maglev_sim.plant.control.gap_pid import get_controller
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.physics.beam import BeamModes
from maglev_sim.plant.physics.magnet import gap_raw, magnetic_force
from maglev_sim.plant.physics.vehicle import vehicle_accelerations
from maglev_sim.plant.state import StateLayout, apply_delta, apply_random_std, layout_from_config


@dataclass
class Kinematics:
    xi: float
    speed: float
    r: float
    w: float
    w_dot: float
    gap_raw: float
    force: float
    a_c: float
    a_m: float
    q_dot: np.ndarray
    p_dot: np.ndarray
    i_raw: float
    i_cmd: float
    i_dot: float
    eta_dot: float
    chi_dot: float
    d_f: float
    saturated: bool
    force_x: float
    force_P: float
    error: float


def _kinematics(
    t: float,
    state: np.ndarray,
    config: Config,
    excitation: Excitation,
    layout: StateLayout | None = None,
    beam: BeamModes | None = None,
) -> Kinematics:
    layout = layout or layout_from_config(config)
    beam = beam or BeamModes.from_config(config.beam)
    u = layout.unpack(state)
    xi = excitation.position(t)
    speed = excitation.speed(t)
    r = excitation.irregularity_value(xi)
    w = beam.displacement(xi, u["q"])
    w_dot = beam.material_velocity(xi, u["q"], u["p"], speed)
    gap = gap_raw(u["z_m"], r, w, config.magnet.delta_0)
    force = magnetic_force(u["i"], gap, config.magnet)
    a_c, a_m = vehicle_accelerations(
        u["z_c"], u["v_c"], u["z_m"], u["v_m"], force, config.vehicle
    )
    x_f, P_ext = excitation.point_force(t)
    q_dot, p_dot = beam.modal_rhs(u["q"], u["p"], force, xi, x_f, P_ext)
    error = gap - config.magnet.delta_0
    ctrl = get_controller(config.controller.name)
    out = ctrl.command_and_derivs(error, u["i"], u["eta"], u["chi"], config)
    i_dot = (out.i_cmd - u["i"]) / config.controller.tau_i
    return Kinematics(
        xi=xi,
        speed=speed,
        r=r,
        w=w,
        w_dot=w_dot,
        gap_raw=gap,
        force=force,
        a_c=a_c,
        a_m=a_m,
        q_dot=q_dot,
        p_dot=p_dot,
        i_raw=out.i_raw,
        i_cmd=out.i_cmd,
        i_dot=i_dot,
        eta_dot=out.eta_dot,
        chi_dot=out.chi_dot,
        d_f=out.d_f,
        saturated=out.saturated,
        force_x=x_f,
        force_P=P_ext,
        error=error,
    )


def equilibrium(
    config: Config,
    position: float | None = None,
    irregularity_value: float | None = None,
) -> NDArray[np.float64]:
    """Closed-form static equilibrium of the truncated modal model (doc 3.6)."""
    config.validate()
    layout = layout_from_config(config)
    beam = BeamModes.from_config(config.beam)
    xi = config.motion.xi if position is None else float(position)
    if irregularity_value is None:
        r = Excitation.from_config(config).irregularity_value(xi)
    else:
        r = float(irregularity_value)
    q = beam.static_modal_amplitudes(config.vehicle.F0, xi)
    p = np.zeros_like(q)
    w = beam.displacement(xi, q)
    z = r + w
    return layout.pack(
        z_c=z,
        v_c=0.0,
        z_m=z,
        v_m=0.0,
        q=q,
        p=p,
        i=config.i0,
        eta=0.0,
        chi=0.0,
    )


def initial_state(config: Config, seed: int | None = None) -> NDArray[np.float64]:
    y0 = equilibrium(config)
    layout = layout_from_config(config)
    if config.initial.delta:
        y0 = apply_delta(y0, layout, config.initial.delta)
    if config.initial.random_std:
        if seed is None:
            raise ValueError("random initial std requires a seed")
        y0 = apply_random_std(y0, layout, config.initial.random_std, seed)
    return y0


def rhs(
    t: float,
    state: np.ndarray,
    config: Config,
    excitation: Excitation,
) -> NDArray[np.float64]:
    """Side-effect free f(S, U; theta). Safe for adaptive retries."""
    layout = layout_from_config(config)
    kin = _kinematics(t, state, config, excitation, layout=layout)
    u = layout.unpack(state)
    return layout.pack(
        z_c=u["v_c"],
        v_c=kin.a_c,
        z_m=u["v_m"],
        v_m=kin.a_m,
        q=kin.q_dot,
        p=kin.p_dot,
        i=kin.i_dot,
        eta=kin.eta_dot,
        chi=kin.chi_dot,
    )


@dataclass
class Observation:
    z_car: float
    z_magnet: float
    v_car: float
    v_magnet: float
    a_car: float
    a_magnet: float
    gap_raw: float
    beam_displacement_at_magnet: float
    beam_velocity_at_magnet: float
    current: float
    current_command: float
    current_raw: float
    magnetic_force: float
    q: np.ndarray
    p: np.ndarray
    irregularity: float
    force_x: float
    force_P: float
    xi: float
    speed: float
    saturated: bool
    error: float


def observe(
    t: float,
    state: np.ndarray,
    config: Config,
    excitation: Excitation,
) -> Observation:
    layout = layout_from_config(config)
    u = layout.unpack(state)
    kin = _kinematics(t, state, config, excitation, layout=layout)
    return Observation(
        z_car=u["z_c"],
        z_magnet=u["z_m"],
        v_car=u["v_c"],
        v_magnet=u["v_m"],
        a_car=kin.a_c,
        a_magnet=kin.a_m,
        gap_raw=kin.gap_raw,
        beam_displacement_at_magnet=kin.w,
        beam_velocity_at_magnet=kin.w_dot,
        current=u["i"],
        current_command=kin.i_cmd,
        current_raw=kin.i_raw,
        magnetic_force=kin.force,
        q=np.array(u["q"], copy=True),
        p=np.array(u["p"], copy=True),
        irregularity=kin.r,
        force_x=kin.force_x,
        force_P=kin.force_P,
        xi=kin.xi,
        speed=kin.speed,
        saturated=kin.saturated,
        error=kin.error,
    )


def equilibrium_residuals(
    config: Config,
    excitation: Excitation | None = None,
) -> np.ndarray:
    y = equilibrium(config)
    exc = excitation or Excitation.from_config(config)
    return rhs(0.0, y, config, exc)
