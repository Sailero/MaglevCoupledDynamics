"""Lumped EMS force F = kappa * i^2 / delta^2."""

from __future__ import annotations

import numpy as np

from maglev_sim.plant.config import MagnetConfig, VehicleConfig


def kappa(magnet: MagnetConfig) -> float:
    return magnet.kappa


def gap_raw(z_m: float, r: float, w: float, delta_0: float) -> float:
    """True air gap. Never clip this value for logging or termination."""
    return delta_0 + z_m - r - w


def magnetic_force(current: float, gap: float, magnet: MagnetConfig) -> float:
    """Attractive force magnitude. Uses delta_floor only to avoid 1/0."""
    safe = gap if gap >= magnet.delta_floor else magnet.delta_floor
    return magnet.kappa * (current * current) / (safe * safe)


def equilibrium_current(vehicle: VehicleConfig, magnet: MagnetConfig) -> float:
    return magnet.delta_0 * np.sqrt(vehicle.F0 / magnet.kappa)
