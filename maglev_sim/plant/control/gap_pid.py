"""Gap PID + derivative filter + first-order current loop (requirements 3.5)."""

from __future__ import annotations

from maglev_sim.plant.config import Config
from maglev_sim.plant.control.base import ControllerOutput


class GapPIDController:
    name = "gap_pid"
    state_names = ("eta", "chi")

    def command_and_derivs(
        self,
        error: float,
        current: float,
        eta: float,
        chi: float,
        config: Config,
    ) -> ControllerOutput:
        del current  # plant current is integrated outside
        ctrl = config.controller
        chi_dot = (error - chi) / ctrl.tau_d
        d_f = chi_dot
        i_raw = config.i0 + ctrl.Kp * error + ctrl.Ki * eta + ctrl.Kd * d_f
        i_cmd = min(max(i_raw, config.magnet.I_min), config.magnet.I_max)
        pushing_high = i_raw > config.magnet.I_max and ctrl.Ki * error > 0.0
        pushing_low = i_raw < config.magnet.I_min and ctrl.Ki * error < 0.0
        eta_dot = 0.0 if (pushing_high or pushing_low) else error
        saturated = i_cmd != i_raw
        return ControllerOutput(
            i_raw=i_raw,
            i_cmd=i_cmd,
            eta_dot=eta_dot,
            chi_dot=chi_dot,
            d_f=d_f,
            saturated=saturated,
        )


def get_controller(name: str) -> GapPIDController:
    if name == "gap_pid":
        return GapPIDController()
    raise ValueError(f"unknown controller {name!r}")
