"""Stationary two-point vehicle--beam dynamics, in SI units.

Displacements and beam modal coordinates are increments from the loaded
equilibrium; the last two states are actual coil currents.  The equations and
column order follow docs/dynamic_model.md, Eqs. (26)--(32).  Numerical defaults
live exclusively in configs/default.yaml.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml


MODEL_VERSION = "stationary_two_point_v2"
Y_names = ["z_c", "z_m1", "z_m2", "w1", "w2"]
U_names = ["P1", "P2"]
O_names = ["gap1", "gap2", "a_m1", "a_m2", "a_c", "i1", "i2", "beam_port_power"]

_SCHEMA = {
    "meta": {"model_version", "parameter_source", "notes", "sources"},
    "geometry": {"positions_m"},
    "vehicle": {"m1_kg", "m2_kg", "m3_kg", "m4_kg", "stiffness_n_m", "damping_ns_m", "gravity"},
    "beam": {"enabled", "n_modes", "length_m", "mass_per_length", "f1_hz", "damping_ratio"},
    "magnet": {"mu0", "turns", "pole_area_m2", "gap_m", "gap_min_m", "gap_max_m", "gap_floor_m", "current_min_a", "current_max_a"},
    "controller": {"kp_a_m", "kd_as_m", "current_gain_v_a"},
    "time": {"dt", "n_steps"},
    "integrator": {"method", "rtol", "scaled_atol", "max_step"},
    "initial": {"offsets", "random_std"},
    "excitation": {"type", "amplitudes_n", "start_s", "duration_s"},
    "dataset": {"name", "n_samples", "seed", "dtype", "plot", "split_fractions", "split_seed", "split_by", "parameter_ranges", "n_parameter_families"},
    "validation": {"model", "solver", "dataset", "scientific", "duration_s", "fit_start_s", "fit_end_s", "gains_v_a", "seed"},
}
_SOURCE_KEYS = {"mechanics", "circuit_and_beam", "parameters", "beam", "positions"}


def _mapping(value: Any, path: str, keys: set[str] | None = None) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a mapping with string keys")
    if keys is not None:
        missing, unknown = keys - value.keys(), value.keys() - keys
        if missing or unknown:
            raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return value


def _number(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{path} must be a finite real number") from error
    if not np.isfinite(result):
        raise ValueError(f"{path} must be a finite real number")
    if positive and result <= 0 or nonnegative and result < 0:
        raise ValueError(f"{path} must be {'positive' if positive else 'nonnegative'}")
    return result


def _integer(value: Any, path: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return int(value)


def _vector(value: Any, path: str, length: int, **limits: bool) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (length,):
        raise ValueError(f"{path} must contain exactly {length} numbers")
    return np.array([_number(item, f"{path}[{index}]", **limits) for index, item in enumerate(value)])


def _boolean(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")


def _state_names(n_modes: int) -> list[str]:
    return ["z_m1", "z_m2", "z_c", "v_m1", "v_m2", "v_c"] + [f"q{n}" for n in range(1, n_modes + 1)] + [f"p{n}" for n in range(1, n_modes + 1)] + ["i1", "i2"]


def validate_config(cfg: dict) -> None:
    """Reject incomplete, unknown, nonfinite, or physically invalid settings.

    Only random standard deviations for inactive beam coordinates (q2/p2 in a
    one-mode run) may be retained and ignored.  Offsets never ignore unknown
    coordinates.  All checks also execute when Python runs with ``-O``.
    """
    _mapping(cfg, "config", set(_SCHEMA))
    for section, keys in _SCHEMA.items():
        _mapping(cfg[section], section, keys)
    meta = cfg["meta"]
    if meta["model_version"] != MODEL_VERSION:
        raise ValueError(f"meta.model_version must be {MODEL_VERSION!r}")
    for key in ("parameter_source", "notes"):
        if not isinstance(meta[key], str):
            raise ValueError(f"meta.{key} must be a string")
    for key, value in _mapping(meta["sources"], "meta.sources", _SOURCE_KEYS).items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"meta.sources.{key} must be a nonempty string")

    vehicle, beam, magnet = cfg["vehicle"], cfg["beam"], cfg["magnet"]
    for key in ("m1_kg", "m2_kg", "m3_kg", "m4_kg", "gravity"):
        _number(vehicle[key], f"vehicle.{key}", positive=True)
    _vector(vehicle["stiffness_n_m"], "vehicle.stiffness_n_m", 2, positive=True)
    _vector(vehicle["damping_ns_m"], "vehicle.damping_ns_m", 2, nonnegative=True)
    _boolean(beam["enabled"], "beam.enabled")
    n_modes = _integer(beam["n_modes"], "beam.n_modes", 1)
    if n_modes not in (1, 2):
        raise ValueError("beam.n_modes must be 1 or 2")
    for key in ("length_m", "mass_per_length", "f1_hz"):
        _number(beam[key], f"beam.{key}", positive=True)
    if isinstance(beam["damping_ratio"], (list, tuple, np.ndarray)):
        _vector(beam["damping_ratio"], "beam.damping_ratio", n_modes, nonnegative=True)
    else:
        _number(beam["damping_ratio"], "beam.damping_ratio", nonnegative=True)
    positions = _vector(cfg["geometry"]["positions_m"], "geometry.positions_m", 2)
    if not 0 < positions[0] < positions[1] < beam["length_m"]:
        raise ValueError("geometry.positions_m must satisfy 0 < a1 < a2 < beam.length_m")
    for key in ("mu0", "pole_area_m2", "gap_m", "gap_min_m", "gap_max_m", "gap_floor_m"):
        _number(magnet[key], f"magnet.{key}", positive=True)
    _integer(magnet["turns"], "magnet.turns", 1)
    _number(magnet["turns"], "magnet.turns", positive=True)
    if not magnet["gap_floor_m"] < magnet["gap_min_m"] < magnet["gap_m"] < magnet["gap_max_m"]:
        raise ValueError("magnet gaps must satisfy floor < minimum < equilibrium < maximum")
    _number(magnet["current_min_a"], "magnet.current_min_a", nonnegative=True)
    _number(magnet["current_max_a"], "magnet.current_max_a", positive=True)
    if magnet["current_max_a"] <= magnet["current_min_a"]:
        raise ValueError("magnet.current_max_a must exceed magnet.current_min_a")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ce = np.float64(magnet["mu0"]) * np.float64(magnet["turns"]) ** 2 * magnet["pole_area_m2"] / 4
        forces = (np.array([vehicle["m1_kg"], vehicle["m2_kg"]]) + (vehicle["m3_kg"] + vehicle["m4_kg"]) / 2) * vehicle["gravity"]
        i0 = magnet["gap_m"] * np.sqrt(forces / ce)
    if not np.isfinite(ce) or ce <= 0:
        raise ValueError("derived magnetic coefficient Ce must be finite and positive")
    if not np.isfinite(i0).all() or not np.all((magnet["current_min_a"] < i0) & (i0 < magnet["current_max_a"])):
        raise ValueError("loaded equilibrium currents must lie strictly inside configured current bounds")
    controller = cfg["controller"]
    for key in ("kp_a_m", "kd_as_m"):
        _number(controller[key], f"controller.{key}", nonnegative=True)
    _number(controller["current_gain_v_a"], "controller.current_gain_v_a", positive=True)

    _number(cfg["time"]["dt"], "time.dt", positive=True)
    _integer(cfg["time"]["n_steps"], "time.n_steps", 2)
    integrator = cfg["integrator"]
    if integrator["method"] != "Radau":
        raise ValueError("integrator.method must be 'Radau'")
    for key in ("rtol", "scaled_atol", "max_step"):
        _number(integrator[key], f"integrator.{key}", positive=True)
    if integrator["rtol"] >= 1:
        raise ValueError("integrator.rtol must be less than one")

    names = set(_state_names(n_modes)) | {"common_m", "differential_m"}
    inactive = {"q2", "p2"} if n_modes == 1 else set()
    for group in ("offsets", "random_std"):
        values = _mapping(cfg["initial"][group], f"initial.{group}")
        allowed = names | (inactive if group == "random_std" else set())
        unknown = values.keys() - allowed
        if unknown:
            raise ValueError(f"initial.{group}: unknown or inactive coordinates {sorted(unknown)}")
        for key, value in values.items():
            _number(value, f"initial.{group}.{key}", nonnegative=(group == "random_std"))

    excitation = cfg["excitation"]
    if excitation["type"] not in ("none", "pulse"):
        raise ValueError("excitation.type must be 'none' or 'pulse'")
    _vector(excitation["amplitudes_n"], "excitation.amplitudes_n", 2)
    _number(excitation["start_s"], "excitation.start_s", nonnegative=True)
    _number(excitation["duration_s"], "excitation.duration_s", positive=True)
    if excitation["type"] == "none" and any(excitation["amplitudes_n"]):
        raise ValueError("excitation.amplitudes_n must be zero when excitation.type is 'none'")

    dataset = cfg["dataset"]
    name = dataset["name"]
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)}
    if not isinstance(name, str) or not name.strip() or any(ord(c) < 32 or c in '/\\:*?<>|"' for c in name) or name.endswith((" ", ".")) or name.split(".")[0].upper() in reserved:
        raise ValueError("dataset.name must be a valid nonempty Windows filename stem")
    for key in ("n_samples", "n_parameter_families"):
        _integer(dataset[key], f"dataset.{key}", 1)
    for key in ("seed", "split_seed"):
        _integer(dataset[key], f"dataset.{key}", 0)
    _boolean(dataset["plot"], "dataset.plot")
    if dataset["dtype"] not in ("float32", "float64"):
        raise ValueError("dataset.dtype must be 'float32' or 'float64'")
    fractions = _vector(dataset["split_fractions"], "dataset.split_fractions", 3, nonnegative=True)
    if abs(float(fractions.sum()) - 1.) > 1e-12:
        raise ValueError("dataset.split_fractions must sum to one")
    if dataset["split_by"] not in ("trajectory", "family"):
        raise ValueError("dataset.split_by must be 'trajectory' or 'family'")
    ranges = _mapping(dataset["parameter_ranges"], "dataset.parameter_ranges")
    for path, limits in ranges.items():
        parts = path.split(".")
        if len(parts) < 2 or parts[0] not in {"geometry", "vehicle", "beam", "magnet", "controller"} or path == "beam.n_modes":
            raise ValueError(f"dataset.parameter_ranges.{path}: expected an existing physical scalar parameter with fixed modal order")
        value: Any = cfg
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, (list, tuple, np.ndarray)) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                raise ValueError(f"dataset.parameter_ranges.{path}: unknown parameter")
        _number(value, f"parameter target {path}")
        bounds = _vector(limits, f"dataset.parameter_ranges.{path}", 2)
        if bounds[0] > bounds[1]:
            raise ValueError(f"dataset.parameter_ranges.{path}: lower bound exceeds upper bound")
        if path == "magnet.turns" and np.ceil(bounds[0]) > np.floor(bounds[1]):
            raise ValueError("dataset.parameter_ranges.magnet.turns must contain at least one integer")

    validation = cfg["validation"]
    for key in ("model", "solver", "dataset", "scientific"):
        _boolean(validation[key], f"validation.{key}")
    _integer(validation["seed"], "validation.seed", 0)
    _number(validation["duration_s"], "validation.duration_s", positive=True)
    _number(validation["fit_start_s"], "validation.fit_start_s", nonnegative=True)
    _number(validation["fit_end_s"], "validation.fit_end_s", positive=True)
    if not validation["fit_start_s"] < validation["fit_end_s"] <= validation["duration_s"]:
        raise ValueError("validation fit interval must satisfy 0 <= start < end <= duration")
    gains = validation["gains_v_a"]
    if not isinstance(gains, (list, tuple)) or not gains:
        raise ValueError("validation.gains_v_a must be a nonempty list")
    for index, gain in enumerate(gains):
        _number(gain, f"validation.gains_v_a[{index}]", positive=True)

    # Finite individual inputs can still overflow the derived physical model.
    # Check the actual combinations before constructing states or integrating.
    try:
        duration = cfg["time"]["dt"] * (cfg["time"]["n_steps"] - 1)
    except OverflowError as error:
        raise ValueError("derived simulation duration must be finite") from error
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        length = np.float64(beam["length_m"])
        mu = np.float64(beam["mass_per_length"])
        omega = 2 * np.pi * np.float64(beam["f1_hz"]) * np.arange(1, n_modes + 1, dtype=float) ** 2
        modal_mass = mu * length / 2
        modal_stiffness = modal_mass * omega ** 2
        ei = mu * (2 * np.pi * np.float64(beam["f1_hz"])) ** 2 * (length / np.pi) ** 4
        inductances = 2 * ce / np.array([magnet["gap_m"], magnet["gap_floor_m"]])
        bandwidth = controller["current_gain_v_a"] / inductances[0]
        modal_damping = 2 * np.asarray(beam["damping_ratio"], dtype=float) * omega
        modal_load = np.sin(np.pi * (positions / length)[:, None] * np.arange(1, n_modes + 1)[None, :]).T @ forces
        static_q = modal_load / modal_stiffness
        coefficients = np.r_[omega, omega ** 2, modal_mass, modal_stiffness, modal_damping, ei, inductances, bandwidth, bandwidth * controller["kp_a_m"], bandwidth * controller["kd_as_m"], static_q, duration, ce, forces]
    if not np.isfinite(coefficients).all() or np.any(omega <= 0) or modal_mass <= 0 or np.any(modal_stiffness <= 0) or ei <= 0 or np.any(inductances <= 0) or bandwidth <= 0:
        raise ValueError("derived frequencies, stiffness, inductance, static state, and simulation duration must be finite and representable")


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loading with duplicate keys reported instead of overwritten."""


def _unique_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("configuration YAML mapping keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path: str | Path) -> dict:
    """Load one complete YAML configuration without filling implicit defaults."""
    with Path(path).open("r", encoding="utf-8") as stream:
        try:
            cfg = yaml.load(stream, Loader=_UniqueSafeLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid configuration YAML: {error}") from error
    validate_config(cfg)
    return cfg


class Model:
    """Pure two-point right-hand side and its equilibrium-based interfaces."""

    def __init__(self, cfg: dict):
        validate_config(cfg)
        self.cfg = deepcopy(cfg)
        vehicle, beam, magnet, controller = (self.cfg[key] for key in ("vehicle", "beam", "magnet", "controller"))
        self.n_modes = beam["n_modes"]
        self.names = _state_names(self.n_modes)
        self.n_state = len(self.names)
        self.q_slice = slice(6, 6 + self.n_modes)
        self.p_slice = slice(6 + self.n_modes, 6 + 2 * self.n_modes)
        self.masses = np.array([vehicle["m1_kg"], vehicle["m2_kg"], vehicle["m3_kg"] + vehicle["m4_kg"]], dtype=float)
        self.stiffness = np.array(vehicle["stiffness_n_m"], dtype=float)
        self.damping = np.array(vehicle["damping_ns_m"], dtype=float)
        mode_indices = np.arange(1, self.n_modes + 1)
        self.phi = np.sin(np.pi * np.array(self.cfg["geometry"]["positions_m"])[:, None] * mode_indices[None, :] / beam["length_m"])
        if not beam["enabled"]:
            self.phi[:] = 0.
        self.omega = 2 * np.pi * beam["f1_hz"] * mode_indices ** 2
        self.modal_mass = np.full(self.n_modes, beam["mass_per_length"] * beam["length_m"] / 2)
        self.zeta = np.broadcast_to(np.asarray(beam["damping_ratio"], dtype=float), (self.n_modes,)).copy()
        self.force_star = (self.masses[:2] + self.masses[2] / 2) * vehicle["gravity"]
        self._ce = magnet["mu0"] * magnet["turns"] ** 2 * magnet["pole_area_m2"] / 4
        self._gap0 = magnet["gap_m"]
        self._gap_floor = magnet["gap_floor_m"]
        self.i0 = self._gap0 * np.sqrt(self.force_star / self._ce)
        self.q_star = self.phi.T @ self.force_star / (self.modal_mass * self.omega ** 2)
        self.w_star = self.phi @ self.q_star
        self.x_star = np.zeros(self.n_state)
        self.x_star[-2:] = self.i0
        self.scales = np.r_[np.full(3, 0.01), np.full(3, 0.5), np.full(self.n_modes, 0.01), np.full(self.n_modes, 0.5), self.i0]
        self._kp, self._kd, self._gain = (controller[key] for key in ("kp_a_m", "kd_as_m", "current_gain_v_a"))
        self._modal_damping = 2 * self.zeta * self.omega
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            finite_linearization = np.isfinite(self.linear_matrix()).all()
            finite_equilibrium = np.isfinite(self.rhs(0., self.x_star)).all()
        if not finite_linearization or not finite_equilibrium:
            raise ValueError("derived mechanical/current dynamics must have finite coefficients and equilibrium derivatives")

    def _state(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.shape != (self.n_state,) or not np.isfinite(values).all():
            raise ValueError(f"state must be a finite vector of length {self.n_state}")
        return values

    @staticmethod
    def _force(force: np.ndarray | None) -> np.ndarray:
        if force is None:
            return np.zeros(2)
        values = np.asarray(force, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("external force must be a finite two-component vector")
        return values

    def gaps(self, x: np.ndarray) -> np.ndarray:
        """Return the raw physical gaps; event detection must use these values."""
        x = self._state(x)
        return self._gap0 + x[:2] - self.phi @ x[self.q_slice]

    def _magnetic_increment(self, x: np.ndarray, gaps: np.ndarray) -> np.ndarray:
        safe_gaps = np.maximum(gaps, self._gap_floor)
        return self._ce * (x[-2:] / safe_gaps) ** 2 - self.force_star

    def rhs(self, t: float, x: np.ndarray, force: np.ndarray | None = None) -> np.ndarray:
        """Evaluate Eq. (26); ``force`` acts downward on the beam's two ports."""
        x, force = self._state(x), self._force(force)
        z, velocity, q, p, currents = x[:3], x[3:6], x[self.q_slice], x[self.p_slice], x[-2:]
        gaps = self._gap0 + z[:2] - self.phi @ q
        safe_gaps = np.maximum(gaps, self._gap_floor)
        delta_force = self._magnetic_increment(x, gaps)
        connection = self.stiffness * (z[:2] - z[2]) + self.damping * (velocity[:2] - velocity[2])
        dx = np.empty(self.n_state)
        dx[:3] = velocity
        dx[3:5] = (-delta_force - connection) / self.masses[:2]
        dx[5] = connection.sum() / self.masses[2]
        dx[self.q_slice] = p
        dx[self.p_slice] = -self._modal_damping * p - self.omega ** 2 * q + self.phi.T @ (delta_force + force) / self.modal_mass
        reference_error = self.i0 + self._kp * (gaps - self._gap0) + self._kd * velocity[:2] - currents
        inductance = 2 * self._ce / safe_gaps
        dx[-2:] = self._gain / inductance * reference_error + currents / safe_gaps * (velocity[:2] - self.phi @ p)
        return dx

    def observations(self, x: np.ndarray, force: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return primary Y and auxiliary O columns in the documented order."""
        x = self._state(x)
        derivative = self.rhs(0., x, force)
        beam_w = self.phi @ x[self.q_slice]
        gaps = self.gaps(x)
        power = float(self._magnetic_increment(x, gaps) @ (self.phi @ x[self.p_slice]))
        return np.r_[x[2], x[:2], beam_w], np.r_[gaps, derivative[3:6], x[-2:], power]

    def initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Add configured increments and seeded common/differential perturbations.

        Currents named i1/i2 are perturbations about i0, just as every other
        named coordinate is perturbed about x_star.  Inactive q2/p2 random
        standard deviations are ignored without consuming a random draw.
        """
        x = self.x_star.copy()
        indices = {name: index for index, name in enumerate(self.names)}
        for group in ("offsets", "random_std"):
            values = self.cfg["initial"][group]
            # Persistent JSON sorts mapping keys. Draw in state order so that
            # replay depends on values and the seed, never YAML key ordering.
            for name in self.names + ["common_m", "differential_m"]:
                if name not in values:
                    continue
                value = values[name]
                increment = float(value) if group == "offsets" else float(rng.normal(0., value))
                if name == "common_m":
                    x[:2] += increment
                elif name == "differential_m":
                    x[:2] += [increment, -increment]
                else:
                    x[indices[name]] += increment
        return x

    def linear_matrix(self) -> np.ndarray:
        """Analytic equilibrium Jacobian, Eqs. (27)--(28)."""
        n = self.n_state
        matrix = np.zeros((n, n))
        matrix[:3, 3:6] = np.eye(3)
        matrix[self.q_slice, self.p_slice] = np.eye(self.n_modes)
        gap_map = np.zeros((2, n))
        gap_map[:, :2] = np.eye(2)
        gap_map[:, self.q_slice] = -self.phi
        force_map = -(2 * self.force_star / self._gap0)[:, None] * gap_map
        force_map[:, -2:] += np.diag(2 * self.force_star / self.i0)
        connection_map = np.zeros((2, n))
        connection_map[:, :2] = np.diag(self.stiffness)
        connection_map[:, 2] = -self.stiffness
        connection_map[:, 3:5] = np.diag(self.damping)
        connection_map[:, 5] = -self.damping
        matrix[3:5] = (-force_map - connection_map) / self.masses[:2, None]
        matrix[5] = connection_map.sum(axis=0) / self.masses[2]
        matrix[self.p_slice] = self.phi.T @ force_map / self.modal_mass[:, None]
        matrix[self.p_slice, self.q_slice] -= np.diag(self.omega ** 2)
        matrix[self.p_slice, self.p_slice] -= np.diag(self._modal_damping)
        bandwidth = self._gain * self._gap0 / (2 * self._ce)
        matrix[-2:] = bandwidth * self._kp * gap_map
        matrix[-2:, 3:5] += np.diag(bandwidth * self._kd + self.i0 / self._gap0)
        matrix[-2:, self.p_slice] -= (self.i0 / self._gap0)[:, None] * self.phi
        matrix[-2:, -2:] -= bandwidth * np.eye(2)
        return matrix

    def rigid_matrix(self) -> np.ndarray:
        """Eight-state closed loop with both beam displacement ports held fixed."""
        indices = np.r_[np.arange(6), self.n_state - 2, self.n_state - 1]
        return self.linear_matrix()[np.ix_(indices, indices)]

    def energy_ports(self, x: np.ndarray, force: np.ndarray | None = None) -> dict[str, float]:
        """Return beam and total mechanical energy rates from Eqs. (29)--(30)."""
        x, force = self._state(x), self._force(force)
        q, p, velocity = x[self.q_slice], x[self.p_slice], x[3:6]
        beam_velocity = self.phi @ p
        delta_force = self._magnetic_increment(x, self.gaps(x))
        magnetic_power = float(delta_force @ beam_velocity)
        external_power = float(force @ beam_velocity)
        beam_dissipation = float(np.sum(self.modal_mass * self._modal_damping * p ** 2))
        mechanical_rate = float(-delta_force @ (velocity[:2] - beam_velocity) - np.dot(self.damping, (velocity[:2] - velocity[2]) ** 2) - beam_dissipation + external_power)
        return {
            "beam_energy": float(0.5 * np.sum(self.modal_mass * (p ** 2 + self.omega ** 2 * q ** 2))),
            "magnetic_power": magnetic_power,
            "external_power": external_power,
            "beam_dissipation": beam_dissipation,
            "beam_energy_rate": magnetic_power + external_power - beam_dissipation,
            "mechanical_energy_rate": mechanical_rate,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_model(cfg: dict) -> dict:
    """Run persistent physical acceptance checks and return measured SI values.

    Checks cover the actual configured model, including the analytic Jacobian,
    loaded equilibrium, modal normalization, free responses, static truncation,
    power projection, and independent energy derivatives.  The complete-beam
    static solution is a truncation reference, never an unqualified equality
    claim for the one/two-mode approximation.
    """
    from scipy.integrate import solve_ivp

    model = Model(cfg)
    equilibrium_rhs = model.rhs(0., model.x_star)
    equilibrium_scaled = float(np.max(np.abs(equilibrium_rhs / model.scales)))
    _require(equilibrium_scaled < 1e-9, "normalized loaded-equilibrium derivative exceeds 1e-9")

    analytic = model.linear_matrix()
    finite_difference = np.empty_like(analytic)
    for index, scale in enumerate(model.scales):
        h = scale * 1e-6
        xp, xm = model.x_star.copy(), model.x_star.copy()
        xp[index] += h
        xm[index] -= h
        finite_difference[:, index] = (model.rhs(0., xp) - model.rhs(0., xm)) / (2 * h)
    scaled_matrix = analytic * model.scales[None, :] / model.scales[:, None]
    jacobian_residual = float(np.max(np.abs((analytic - finite_difference) * model.scales[None, :] / model.scales[:, None])) / max(1., np.max(np.abs(scaled_matrix))))
    _require(jacobian_residual < 1e-7, "analytic Jacobian disagrees with the nonlinear differential")

    beam = cfg["beam"]
    positions = np.asarray(cfg["geometry"]["positions_m"])
    length = beam["length_m"]
    ei = beam["mass_per_length"] * (2 * np.pi * beam["f1_hz"]) ** 2 * (length / np.pi) ** 4
    static_truncated = np.zeros(2)
    if beam["enabled"]:
        for mode in range(1, model.n_modes + 1):
            shapes = np.sin(mode * np.pi * positions / length)
            static_truncated += shapes * np.dot(shapes, model.force_star) / ((beam["mass_per_length"] * length / 2) * (2 * np.pi * beam["f1_hz"] * mode ** 2) ** 2)
    static_residual = float(np.max(np.abs(model.w_star - static_truncated)))
    _require(static_residual <= 1e-12 * max(1., np.max(np.abs(static_truncated))), "loaded modal static deflection disagrees with same-order analytical truncation")
    static_full = np.zeros(2)
    if beam["enabled"]:
        for row, x in enumerate(positions):
            for a, force in zip(positions, model.force_star):
                b = length - a
                if x <= a:
                    static_full[row] += force * b * x * (length ** 2 - b ** 2 - x ** 2) / (6 * ei * length)
                else:
                    static_full[row] += force * a * (length - x) * (length ** 2 - a ** 2 - (length - x) ** 2) / (6 * ei * length)

    free_cfg = deepcopy(cfg)
    free_cfg["beam"]["enabled"] = False
    free_model = Model(free_cfg)
    free_residuals = []
    for mode in range(model.n_modes):
        x0 = free_model.x_star.copy()
        amplitude = model.scales[model.q_slice.start + mode] * 1e-3
        x0[model.q_slice.start + mode] = amplitude
        omega, zeta = model.omega[mode], model.zeta[mode]
        times = np.linspace(0., 2 * np.pi / omega, 41)
        integration = cfg["integrator"]
        result = solve_ivp(free_model.rhs, (0., times[-1]), x0, t_eval=times,
                           method=integration["method"], rtol=integration["rtol"],
                           atol=integration["scaled_atol"] * free_model.scales,
                           max_step=integration["max_step"])
        _require(result.success, f"free mode {mode + 1} integration failed")
        if zeta < 1:
            wd = omega * np.sqrt(1 - zeta ** 2)
            exact = amplitude * np.exp(-zeta * omega * times) * (np.cos(wd * times) + zeta * omega / wd * np.sin(wd * times))
        elif zeta == 1:
            exact = amplitude * np.exp(-omega * times) * (1 + omega * times)
        else:
            root = np.sqrt(zeta ** 2 - 1)
            r1, r2 = -omega / (zeta + root), -omega * (zeta + root)
            exact = amplitude * (r2 * np.exp(r1 * times) - r1 * np.exp(r2 * times)) / (r2 - r1)
        residual = float(np.max(np.abs(result.y[model.q_slice.start + mode] - exact)))
        _require(residual / amplitude < 1e-6, f"free mode {mode + 1} disagrees with its analytical response")
        free_residuals.append(residual)

    rng = np.random.default_rng(cfg["validation"]["seed"])
    port_residuals, beam_residuals, mechanical_residuals = [], [], []
    force = np.array([1., -0.7]) * np.min(model.force_star) * 1e-3
    for _ in range(8):
        x = model.x_star + rng.normal(size=model.n_state) * model.scales * 1e-3
        delta_force = model._magnetic_increment(x, model.gaps(x))
        p = x[model.p_slice]
        physical_power = float(delta_force @ (model.phi @ p))
        modal_power = float((model.phi.T @ delta_force) @ p)
        port_residuals.append(abs(physical_power - modal_power) / max(abs(physical_power), abs(modal_power), 1e-15))
        dx = model.rhs(0., x, force)
        ports = model.energy_ports(x, force)
        beam_derivative = float(np.sum(model.modal_mass * (p * dx[model.p_slice] + model.omega ** 2 * x[model.q_slice] * p)))
        vehicle_derivative = float(np.dot(model.masses * x[3:6], dx[3:6]) + np.dot(model.stiffness * (x[:2] - x[2]), x[3:5] - x[5]))
        mechanical_derivative = beam_derivative + vehicle_derivative
        beam_norm = max(abs(beam_derivative), abs(ports["magnetic_power"]), abs(ports["external_power"]), abs(ports["beam_dissipation"]), 1e-15)
        mechanical_norm = max(abs(mechanical_derivative), abs(beam_derivative), abs(vehicle_derivative), abs(ports["mechanical_energy_rate"]), 1e-15)
        beam_residuals.append(abs(beam_derivative - ports["beam_energy_rate"]) / beam_norm)
        mechanical_residuals.append(abs(mechanical_derivative - ports["mechanical_energy_rate"]) / mechanical_norm)
    _require(max(port_residuals) < 1e-10, "point/modal port-power identity residual exceeds 1e-10")
    _require(max(beam_residuals) < 1e-10, "beam-energy derivative identity residual exceeds 1e-10")
    _require(max(mechanical_residuals) < 1e-10, "mechanical-energy derivative identity residual exceeds 1e-10")
    canonical_cfg = deepcopy(cfg)
    for group in ("offsets", "random_std"):
        canonical_cfg["initial"][group] = dict(sorted(cfg["initial"][group].items()))
    original_initial = model.initial_state(np.random.default_rng(cfg["validation"]["seed"]))
    replay_initial = Model(canonical_cfg).initial_state(np.random.default_rng(cfg["validation"]["seed"]))
    _require(np.array_equal(original_initial, replay_initial), "initial replay must be independent of mapping order")
    return {
        "passed": True,
        "model_version": MODEL_VERSION,
        "n_modes": model.n_modes,
        "n_state": model.n_state,
        "equilibrium_scaled_max": equilibrium_scaled,
        "equilibrium_rhs_si": equilibrium_rhs.tolist(),
        "force_star_n": model.force_star.tolist(),
        "i0_a": model.i0.tolist(),
        "q_star_m": model.q_star.tolist(),
        "w_star_m": model.w_star.tolist(),
        "beam_ei_n_m2": float(ei),
        "same_order_static_w_m": static_truncated.tolist(),
        "same_order_static_max_error_m": static_residual,
        "full_beam_static_w_m": static_full.tolist(),
        "static_truncation_error_m": (model.w_star - static_full).tolist(),
        "free_modal_max_errors_m": free_residuals,
        "jacobian_scaled_relative_residual": jacobian_residual,
        "port_power_relative_residual": float(max(port_residuals)),
        "beam_energy_relative_residual": float(max(beam_residuals)),
        "mechanical_energy_relative_residual": float(max(mechanical_residuals)),
        "initial_mapping_order_independent": True,
        "inactive_random_modal_coordinates": [name for name in cfg["initial"]["random_std"] if name not in model.names and name not in {"common_m", "differential_m"}],
    }
