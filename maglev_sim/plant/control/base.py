"""Replaceable levitation controller interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from maglev_sim.plant.config import Config


@dataclass
class ControllerOutput:
    i_raw: float
    i_cmd: float
    eta_dot: float
    chi_dot: float
    d_f: float
    saturated: bool


class Controller(Protocol):
    name: str
    state_names: tuple[str, ...]

    def command_and_derivs(
        self,
        error: float,
        current: float,
        eta: float,
        chi: float,
        config: Config,
    ) -> ControllerOutput: ...
