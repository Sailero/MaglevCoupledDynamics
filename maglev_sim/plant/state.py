"""State vector layout S = [zc, vc, zm, vm, q1, p1, ..., qN, pN, i, eta, chi]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maglev_sim.plant.config import Config

STATE_UNITS = {
    "z_c": "m",
    "v_c": "m/s",
    "z_m": "m",
    "v_m": "m/s",
    "q": "m",
    "p": "m/s",
    "i": "A",
    "eta": "m s",
    "chi": "m",
}


@dataclass(frozen=True)
class StateLayout:
    n_modes: int

    @property
    def dim(self) -> int:
        return 7 + 2 * self.n_modes

    @property
    def z_c(self) -> int:
        return 0

    @property
    def v_c(self) -> int:
        return 1

    @property
    def z_m(self) -> int:
        return 2

    @property
    def v_m(self) -> int:
        return 3

    @property
    def i(self) -> int:
        return 4 + 2 * self.n_modes

    @property
    def eta(self) -> int:
        return 5 + 2 * self.n_modes

    @property
    def chi(self) -> int:
        return 6 + 2 * self.n_modes

    def q_index(self, j: int) -> int:
        """0-based mode index j."""
        return 4 + 2 * j

    def p_index(self, j: int) -> int:
        return 5 + 2 * j

    @property
    def names(self) -> list[str]:
        names = ["z_c", "v_c", "z_m", "v_m"]
        for j in range(1, self.n_modes + 1):
            names.extend([f"q_{j}", f"p_{j}"])
        names.extend(["i", "eta", "chi"])
        return names

    def unpack(self, state: np.ndarray) -> dict[str, np.ndarray | float]:
        y = np.asarray(state, dtype=float).reshape(-1)
        if y.size != self.dim:
            raise ValueError(f"expected state dim {self.dim}, got {y.size}")
        q = np.empty(self.n_modes)
        p = np.empty(self.n_modes)
        for j in range(self.n_modes):
            q[j] = y[self.q_index(j)]
            p[j] = y[self.p_index(j)]
        return {
            "z_c": float(y[self.z_c]),
            "v_c": float(y[self.v_c]),
            "z_m": float(y[self.z_m]),
            "v_m": float(y[self.v_m]),
            "q": q,
            "p": p,
            "i": float(y[self.i]),
            "eta": float(y[self.eta]),
            "chi": float(y[self.chi]),
        }

    def pack(
        self,
        *,
        z_c: float,
        v_c: float,
        z_m: float,
        v_m: float,
        q: np.ndarray,
        p: np.ndarray,
        i: float,
        eta: float,
        chi: float,
    ) -> np.ndarray:
        y = np.empty(self.dim, dtype=float)
        y[self.z_c] = z_c
        y[self.v_c] = v_c
        y[self.z_m] = z_m
        y[self.v_m] = v_m
        q = np.asarray(q, dtype=float).reshape(-1)
        p = np.asarray(p, dtype=float).reshape(-1)
        if q.size != self.n_modes or p.size != self.n_modes:
            raise ValueError("q/p length must equal n_modes")
        for j in range(self.n_modes):
            y[self.q_index(j)] = q[j]
            y[self.p_index(j)] = p[j]
        y[self.i] = i
        y[self.eta] = eta
        y[self.chi] = chi
        return y


def layout_from_config(config: Config) -> StateLayout:
    return StateLayout(n_modes=int(config.beam.n_modes))


def active_indices(config: Config) -> list[int]:
    """Indices that actually evolve. Frozen beam modes are omitted when w is off."""
    layout = layout_from_config(config)
    idx = [layout.z_c, layout.v_c, layout.z_m, layout.v_m]
    if config.beam.enabled:
        for j in range(layout.n_modes):
            idx.extend([layout.q_index(j), layout.p_index(j)])
    idx.extend([layout.i, layout.eta, layout.chi])
    return idx


def name_index(layout: StateLayout) -> dict[str, int]:
    return {name: k for k, name in enumerate(layout.names)}


def apply_delta(state: np.ndarray, layout: StateLayout, delta: dict[str, float]) -> np.ndarray:
    y = np.array(state, dtype=float, copy=True)
    index = name_index(layout)
    for key, value in delta.items():
        if key not in index:
            raise KeyError(f"unknown state name {key!r}")
        y[index[key]] += float(value)
    return y


def apply_random_std(
    state: np.ndarray,
    layout: StateLayout,
    std: dict[str, float],
    seed: int,
) -> np.ndarray:
    y = np.array(state, dtype=float, copy=True)
    index = name_index(layout)
    rng = np.random.default_rng(seed)
    for key, sigma in std.items():
        if key not in index:
            raise KeyError(f"unknown state name {key!r}")
        y[index[key]] += float(sigma) * rng.standard_normal()
    return y
