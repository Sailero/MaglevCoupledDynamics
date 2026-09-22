"""Two-mass vertical vehicle in increment coordinates (no extra gravity)."""

from __future__ import annotations

from maglev_sim.plant.config import VehicleConfig


def vehicle_accelerations(
    z_c: float,
    v_c: float,
    z_m: float,
    v_m: float,
    force: float,
    vehicle: VehicleConfig,
) -> tuple[float, float]:
    """Return (a_c, a_m). Downward positive. Magnet feels -(F - F0)."""
    susp = vehicle.k_s * (z_c - z_m) + vehicle.c_s * (v_c - v_m)
    a_c = -susp / vehicle.m_c
    a_m = (susp - (force - vehicle.F0)) / vehicle.m_m
    return a_c, a_m
