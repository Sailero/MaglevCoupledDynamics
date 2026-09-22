"""Simply-supported Euler–Bernoulli beam, analytical sine modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maglev_sim.plant.config import BeamConfig


def mode_mass(mu_b: float, length: float) -> float:
    return 0.5 * mu_b * length


def natural_frequency(j: int, beam: BeamConfig) -> float:
    """Circular frequency omega_j for 1-based mode index j."""
    return (j * np.pi / beam.L) ** 2 * np.sqrt(beam.E * beam.I_b / beam.mu_b)


def phi(j: int, x: float, length: float) -> float:
    return float(np.sin(j * np.pi * x / length))


def phi_prime(j: int, x: float, length: float) -> float:
    return float((j * np.pi / length) * np.cos(j * np.pi * x / length))


def static_midspan_point_load(P: float, beam: BeamConfig) -> float:
    return P * beam.L**3 / (48.0 * beam.E * beam.I_b)


@dataclass
class BeamModes:
    n_modes: int
    L: float
    M: np.ndarray
    omega: np.ndarray
    zeta: np.ndarray
    enabled: bool

    @classmethod
    def from_config(cls, beam: BeamConfig) -> BeamModes:
        n = int(beam.n_modes)
        j = np.arange(1, n + 1, dtype=float)
        M = np.full(n, mode_mass(beam.mu_b, beam.L))
        omega = (j * np.pi / beam.L) ** 2 * np.sqrt(beam.E * beam.I_b / beam.mu_b)
        zeta = np.array([beam.zeta_j(int(jj)) for jj in j], dtype=float)
        return cls(
            n_modes=n,
            L=beam.L,
            M=M,
            omega=omega,
            zeta=zeta,
            enabled=bool(beam.enabled),
        )

    def shape(self, x: float) -> np.ndarray:
        if self.n_modes == 0:
            return np.zeros(0)
        j = np.arange(1, self.n_modes + 1, dtype=float)
        return np.sin(j * np.pi * x / self.L)

    def shape_prime(self, x: float) -> np.ndarray:
        if self.n_modes == 0:
            return np.zeros(0)
        j = np.arange(1, self.n_modes + 1, dtype=float)
        return (j * np.pi / self.L) * np.cos(j * np.pi * x / self.L)

    def displacement(self, x: float, q: np.ndarray) -> float:
        if not self.enabled or self.n_modes == 0:
            return 0.0
        return float(self.shape(x) @ q)

    def material_velocity(
        self,
        x: float,
        q: np.ndarray,
        p: np.ndarray,
        speed: float,
    ) -> float:
        """d/dt w(xi(t), t) = Σ φ_j p_j + v Σ φ'_j q_j."""
        if not self.enabled or self.n_modes == 0:
            return 0.0
        return float(self.shape(x) @ p + speed * (self.shape_prime(x) @ q))

    def modal_rhs(
        self,
        q: np.ndarray,
        p: np.ndarray,
        force_at_magnet: float,
        xi: float,
        x_f: float,
        P_ext: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_dot, p_dot). Loads the full magnetic force F, not F-F0."""
        n = self.n_modes
        if n == 0:
            return np.zeros(0), np.zeros(0)
        if not self.enabled:
            return np.zeros(n), np.zeros(n)
        phi_xi = self.shape(xi)
        phi_f = self.shape(x_f)
        q_dot = np.asarray(p, dtype=float).copy()
        p_dot = (
            -2.0 * self.zeta * self.omega * p
            - self.omega**2 * q
            + (phi_xi * force_at_magnet + phi_f * P_ext) / self.M
        )
        return q_dot, p_dot

    def static_modal_amplitudes(self, force: float, xi: float) -> np.ndarray:
        if self.n_modes == 0 or not self.enabled:
            return np.zeros(self.n_modes)
        phi_xi = self.shape(xi)
        return phi_xi * force / (self.M * self.omega**2)
