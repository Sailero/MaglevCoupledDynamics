"""Radau integration, physical termination, and reproducible numerical checks."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.integrate import Radau
from scipy.optimize import brentq


def external_force(t: float, cfg: dict) -> np.ndarray:
    """Known downward beam forces; pulse endpoints have zero force and slope."""
    excitation = cfg["excitation"]
    if excitation["type"] == "none":
        return np.zeros(2)
    if excitation["type"] != "pulse":
        raise ValueError("excitation.type must be none or pulse")
    start, duration = excitation["start_s"], excitation["duration_s"]
    if not start < t < start + duration:
        return np.zeros(2)
    return np.asarray(excitation["amplitudes_n"], dtype=float) * np.sin(np.pi * (t - start) / duration) ** 2


def _bounds(model, x):
    """Signed physical distances to limits, independent of division safeguards."""
    magnet = model.cfg["magnet"]
    gaps = model.gaps(x)
    values, names = [], []
    for kind, actual, lower, upper in (
        ("gap", gaps, magnet["gap_min_m"], magnet["gap_max_m"]),
        ("current", x[-2:], magnet["current_min_a"], magnet["current_max_a"]),
    ):
        for j in range(2):
            values.extend((actual[j] - lower, upper - actual[j]))
            names.extend((f"{kind}{j + 1}_min", f"{kind}{j + 1}_max"))
    return np.asarray(values), names


def simulate(model, x0: np.ndarray) -> dict:
    """Return every requested sample and a complete terminal state, even on failure.

    Low-level Radau stepping retains the last *accepted* state if an RHS or
    Newton solve fails. Dense output is read only on the configured time grid.
    Boundary roots are located within accepted steps and saved separately.
    """
    cfg = model.cfg
    t = cfg["time"]["dt"] * np.arange(cfg["time"]["n_steps"], dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (model.n_state,):
        raise ValueError(f"x0 must have shape ({model.n_state},)")
    result = {
        "t": t,
        "X": np.full((len(t), model.n_state), np.nan),
        "Y": np.full((len(t), 5), np.nan),
        "U": np.full((len(t), 2), np.nan),
        "O": np.full((len(t), 8), np.nan),
        "valid": np.zeros(len(t), dtype=bool),
        "termination_reason": "completed",
        "termination_time": 0.0,
        "termination_state": x0.copy(),
        "solver_message": "Reached the requested final time.",
        "success": True,
    }

    def finish(reason, time, state, message):
        result.update(termination_reason=reason, termination_time=float(time),
                      termination_state=np.asarray(state).copy(), solver_message=str(message),
                      success=reason == "completed")
        return result

    def save(index, state):
        force = external_force(t[index], cfg)
        y, observations = model.observations(state, force)
        if np.shape(y) != (5,) or np.shape(observations) != (8,):
            raise ValueError("Observation columns must have shapes (5,) and (8,)")
        if not all(np.isfinite(v).all() for v in (state, force, y, observations)):
            raise FloatingPointError("Nonfinite state or observation")
        result["X"][index], result["Y"][index] = state, y
        result["U"][index], result["O"][index] = force, observations
        result["valid"][index] = True

    if not np.isfinite(x0).all():
        return finish("solver_failure", 0.0, x0, "Nonfinite initial state")
    initial_bounds, names = _bounds(model, x0)
    # Roundoff at an exactly represented physical boundary is not an overshoot.
    tol = 32 * np.finfo(float).eps * np.maximum(1.0, abs(initial_bounds))
    outside = np.flatnonzero(initial_bounds < -tol)
    if len(outside):
        return finish(names[outside[0]], 0.0, x0, "Initial state exceeds a physical limit")
    try:
        save(0, x0)
    except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return finish("solver_failure", 0.0, x0, exc)
    on_boundary = np.flatnonzero(initial_bounds <= tol)
    if len(on_boundary):
        return finish(names[on_boundary[0]], 0.0, x0, "Initial state is on a physical limit")
    if len(t) == 1:
        return result

    settings = cfg["integrator"]
    if settings["method"] != "Radau":
        raise ValueError("Only the Radau integrator is supported")
    segments = [0.0, float(t[-1])]
    pulse = cfg["excitation"]
    if pulse["type"] == "pulse":
        for boundary in (pulse["start_s"], pulse["start_s"] + pulse["duration_s"]):
            if 0 < boundary < t[-1]:
                segments.append(float(boundary))
    segments = sorted(set(segments))
    accepted_t, accepted_x, next_sample = 0.0, x0.copy(), 1

    def rhs(time, state):
        derivative = model.rhs(time, state, external_force(time, cfg))
        if not np.isfinite(derivative).all():
            raise FloatingPointError("Nonfinite right-hand side")
        return derivative

    for start, end in zip(segments[:-1], segments[1:]):
        max_step = settings["max_step"]
        if pulse["type"] == "pulse" and pulse["start_s"] <= start < pulse["start_s"] + pulse["duration_s"]:
            # Without this local bound a very narrow, smooth pulse can fall
            # between every RHS evaluation of an otherwise quiet trajectory.
            max_step = min(max_step, pulse["duration_s"] / 8)
        try:
            solver = Radau(rhs, start, accepted_x.copy(), end,
                           rtol=settings["rtol"],
                           atol=settings["scaled_atol"] * np.asarray(model.scales),
                           max_step=max_step)
            while solver.status == "running":
                old_t, old_x = accepted_t, accepted_x
                message = solver.step()
                if solver.status == "failed":
                    return finish("solver_failure", old_t, old_x, message)
                dense = solver.dense_output()
                new_t, new_x = float(solver.t), solver.y.copy()
                if not np.isfinite(new_x).all():
                    return finish("solver_failure", old_t, old_x, "Nonfinite accepted state")
                accepted_t, accepted_x = new_t, new_x
                before, _ = _bounds(model, old_x)
                after, _ = _bounds(model, new_x)
                crossings = np.flatnonzero((before > 0) & (after <= 0))
                roots = []
                for boundary in crossings:
                    def boundary_value(time):
                        if time == old_t:
                            return before[boundary]
                        if time == new_t:
                            return after[boundary]
                        return _bounds(model, dense(time))[0][boundary]
                    root = brentq(boundary_value,
                                  old_t, new_t, xtol=5e-15, rtol=1e-14)
                    roots.append((root, boundary))
                event = min(roots) if roots else None
                stop_t = event[0] if event else new_t
                # Radau has accepted this step already. If an observation
                # fails while reading dense samples, retain this state (or the
                # earlier physical root) rather than the previous step.
                if event:
                    accepted_t, accepted_x = stop_t, dense(stop_t).copy()
                sample_tolerance = 16 * np.finfo(float).eps * max(1.0, abs(stop_t)) if event else 0.0
                while next_sample < len(t) and t[next_sample] <= stop_t + sample_tolerance:
                    save(next_sample, dense(t[next_sample]))
                    next_sample += 1
                if event:
                    state = dense(stop_t)
                    return finish(names[event[1]], stop_t, state, "Reached a physical limit")
        except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return finish("solver_failure", accepted_t, accepted_x, f"{type(exc).__name__}: {exc}")
    return finish("completed", accepted_t, accepted_x, "Reached the requested final time.")


def _require(condition, message="Numerical acceptance check failed"):
    """Acceptance thresholds remain active with Python optimization enabled."""
    if not condition:
        raise AssertionError(message)


def validate_solver(cfg: dict) -> dict:
    """Check exact event witnesses, failure retention, sampling, and real dynamics."""
    from .model import Model

    test_cfg = deepcopy(cfg)
    test_cfg["time"] = {"dt": cfg["time"]["dt"], "n_steps": 11}
    dt = test_cfg["time"]["dt"]
    test_cfg["excitation"].update(type="none", amplitudes_n=[0.0, 0.0])

    class Probe:
        # Analytical witnesses deliberately use no magnetic-force model: event
        # localization is checked against independently known crossing times.
        def __init__(self, configuration):
            self.cfg = deepcopy(configuration)
            self.n_state, self.scales = 10, np.ones(10)
            self.x_star = np.zeros(10)
            self.x_star[-2:] = (configuration["magnet"]["current_min_a"] + configuration["magnet"]["current_max_a"]) / 2
            self.rate, self.failure_after, self.integrate_force = np.zeros(10), None, False

        def gaps(self, state):
            return self.cfg["magnet"]["gap_m"] + state[:2]

        def rhs(self, time, state, force=None):
            if self.failure_after is not None and time > self.failure_after:
                raise FloatingPointError("Deliberate failure for accepted-state verification")
            derivative = self.rate.copy()
            if self.integrate_force:
                derivative[6] = force[0]
            return derivative

        def observations(self, state, force=None):
            return np.r_[state[2], state[:2], state[6], state[6]], np.r_[self.gaps(state), np.zeros(3), state[-2:], 0.0]

    probe = Probe(test_cfg)
    baseline = simulate(probe, probe.x_star)
    _require((baseline["success"] and baseline["valid"].all()), "Numerical acceptance check failed")
    _require((np.array_equal(baseline["t"], dt * np.arange(11))), "Numerical acceptance check failed")
    event_report = []
    for index in range(8):
        probe = Probe(test_cfg)
        start_bounds, labels = _bounds(probe, probe.x_star)
        coordinate = index // 2 if index < 4 else 8 + (index - 4) // 2
        event_time = 3.25 * dt
        probe.rate[coordinate] = (-1 if index % 2 == 0 else 1) * start_bounds[index] / event_time
        run = simulate(probe, probe.x_star)
        _require((run["termination_reason"] == labels[index]), run["solver_message"])
        time_error = abs(run["termination_time"] - event_time)
        state_error = float(np.max(abs(run["termination_state"] - probe.x_star - event_time * probe.rate)))
        _require((time_error < 1e-10 and state_error < 1e-8), "Numerical acceptance check failed")
        _require((np.array_equal(run["valid"], run["t"] <= event_time)), "Numerical acceptance check failed")
        for name in ("X", "Y", "U", "O"):
            _require((np.isfinite(run[name][run["valid"]]).all()), "Numerical acceptance check failed")
            _require((np.isnan(run[name][~run["valid"]]).all()), "Numerical acceptance check failed")
        event_report.append({"reason": labels[index], "time_s": run["termination_time"],
                             "time_error_s": time_error, "state_max_error": state_error})
    probe = Probe(test_cfg)
    outside = probe.x_star.copy()
    outside[0] = test_cfg["magnet"]["gap_min_m"] - test_cfg["magnet"]["gap_m"] - 0.001
    run = simulate(probe, outside)
    _require((not run["valid"].any() and run["termination_time"] == 0), "Numerical acceptance check failed")
    _require((np.array_equal(run["termination_state"], outside)), "Numerical acceptance check failed")
    boundary = probe.x_star.copy()
    boundary[0] = test_cfg["magnet"]["gap_min_m"] - test_cfg["magnet"]["gap_m"]
    boundary_run = simulate(probe, boundary)
    _require((boundary_run["valid"][0] and boundary_run["valid"].sum() == 1), "Numerical acceptance check failed")
    probe.failure_after = 4.7 * dt
    failed = simulate(probe, probe.x_star)
    _require((failed["termination_reason"] == "solver_failure"), "Numerical acceptance check failed")
    _require((0 < failed["termination_time"] < probe.failure_after), "Numerical acceptance check failed")
    _require((np.array_equal(failed["termination_state"], probe.x_star)), "Numerical acceptance check failed")

    pulse_cfg = deepcopy(test_cfg)
    pulse_cfg["excitation"].update(type="pulse", start_s=4.25 * dt,
                                   duration_s=dt / 100, amplitudes_n=[12.0, -3.0])
    pulse_cfg["integrator"]["max_step"] = 10 * dt
    probe = Probe(pulse_cfg)
    probe.integrate_force = True
    pulse_run = simulate(probe, probe.x_star)
    expected_area = 12 * pulse_cfg["excitation"]["duration_s"] / 2
    area_error = float(abs(pulse_run["X"][-1, 6] - expected_area))
    _require((pulse_run["success"] and area_error < max(1e-10, expected_area * 1e-5)), "Numerical acceptance check failed")

    real_cfg = deepcopy(test_cfg)
    model = Model(real_cfg)
    equilibrium_run = simulate(model, model.x_star)
    equilibrium_error = float(np.max(abs((equilibrium_run["X"] - model.x_star) / model.scales)))
    _require((equilibrium_run["success"] and equilibrium_error < 1e-9), "Numerical acceptance check failed")
    initial = model.initial_state(np.random.default_rng(cfg["validation"]["seed"]))
    first, second = simulate(model, initial), simulate(model, initial)
    _require((np.array_equal(first["X"], second["X"], equal_nan=True)), "Numerical acceptance check failed")
    _require((first["termination_reason"] == second["termination_reason"]), "Numerical acceptance check failed")
    return {"passed": True, "events": event_report,
            "outside_initial_all_invalid": True, "boundary_initial_valid": True,
            "failure_retained_time_s": failed["termination_time"],
            "narrow_pulse_duration_s": pulse_cfg["excitation"]["duration_s"],
            "narrow_pulse_area_error_n_s": area_error,
            "equilibrium_scaled_max_error": equilibrium_error, "repeat_exact": True}


def _spectrum(matrix):
    values = np.linalg.eigvals(matrix)
    order = np.lexsort((values.imag, values.real))
    return [[float(value.real), float(value.imag)] for value in values[order]]


def _mode_fit(model, run, fit_start, fit_end):
    """Project nonlinear increments onto independently computed linear modes."""
    scales = np.asarray(model.scales)
    matrix = model.linear_matrix() * scales[None, :] / scales[:, None]
    values, vectors = np.linalg.eig(matrix)
    candidates = np.flatnonzero(values.imag > 1e-8)
    dominant = candidates[np.argmax(values[candidates].real)]
    eigenvalue = values[dominant]
    coefficient = ((run["X"] - model.x_star) / scales) @ np.linalg.inv(vectors)[dominant]
    mask = run["valid"] & (run["t"] >= fit_start) & (run["t"] <= fit_end)
    time, signal = run["t"][mask], coefficient[mask]
    _require((len(time) >= 10 and np.min(abs(signal)) > 0), "Numerical acceptance check failed")
    log_amplitude, phase = np.log(abs(signal)), np.unwrap(np.angle(signal))
    growth = float(np.polyfit(time, log_amplitude, 1)[0])
    frequency = float(abs(np.polyfit(time, phase, 1)[0]) / (2 * np.pi))
    alpha_error, frequency_error = abs(growth - eigenvalue.real), abs(frequency - eigenvalue.imag / (2 * np.pi))
    _require((alpha_error < max(0.05 * abs(eigenvalue.real), 1e-3)), (growth, eigenvalue))
    _require((frequency_error < 0.01 * abs(eigenvalue.imag / (2 * np.pi))), (frequency, eigenvalue))
    from scipy.signal import find_peaks

    ordinary = run["X"][mask, model.q_slice.start]
    peaks, _ = find_peaks(abs(ordinary))
    ordinary_fit = None
    if len(peaks) >= 4:
        ordinary_fit = {"signal": "q1_absolute_peaks", "n_peaks": int(len(peaks)),
                        "alpha_per_s": float(np.polyfit(time[peaks], np.log(abs(ordinary[peaks])), 1)[0]),
                        "frequency_hz": float(1 / (2 * np.mean(np.diff(time[peaks]))))}
    return {"linear_eigenvalue": [float(eigenvalue.real), float(eigenvalue.imag)],
            "linear_frequency_hz": float(eigenvalue.imag / (2 * np.pi)),
            "projected_alpha_per_s": growth, "projected_frequency_hz": frequency,
            "alpha_absolute_error_per_s": float(alpha_error),
            "frequency_absolute_error_hz": float(frequency_error),
            "ordinary_signal_fit": ordinary_fit,
            "projection_eigenvector_condition": float(np.linalg.cond(vectors)),
            "fit_start_s": float(time[0]), "fit_end_s": float(time[-1]), "fit_n_samples": int(len(time))}


def _energy_check(model, run, fit_start, fit_end):
    """Compare actual powers, state derivatives, and integrated beam energy."""
    from scipy.integrate import simpson

    states, time = run["X"][run["valid"]], run["t"][run["valid"]]
    q, p = states[:, model.q_slice], states[:, model.p_slice]
    beam_velocity = p @ model.phi.T
    magnet = model.cfg["magnet"]
    coefficient = magnet["mu0"] * magnet["turns"] ** 2 * magnet["pole_area_m2"] / 4
    gaps = np.array([model.gaps(state) for state in states])
    force_increment = coefficient * states[:, -2:] ** 2 / gaps ** 2 - model.force_star
    force = run["U"][run["valid"]]
    magnetic = np.sum(force_increment * beam_velocity, axis=1)
    modal = np.sum((force_increment @ model.phi) * p, axis=1)
    external = np.sum(force * beam_velocity, axis=1)
    dissipation = np.sum(2 * model.zeta * model.omega * model.modal_mass * p ** 2, axis=1)
    energy = 0.5 * np.sum(model.modal_mass * (p ** 2 + model.omega ** 2 * q ** 2), axis=1)
    derivatives = np.array([model.rhs(ti, state, fi) for ti, state, fi in zip(time, states, force)])
    gradient_rate = np.sum(model.modal_mass * (p * derivatives[:, model.p_slice] + model.omega ** 2 * q * p), axis=1)
    net_power = magnetic + external - dissipation
    ports = [model.energy_ports(state, fi) for state, fi in zip(states, force)]
    reported_rate = np.array([port["beam_energy_rate"] for port in ports])
    relative_scale = max(float(np.max(abs(magnetic))), float(np.max(abs(dissipation))), np.finfo(float).tiny)
    port_residual = float(np.max(abs(magnetic - modal)))
    derivative_residual = float(np.max(abs(gradient_rate - net_power)))
    reported_residual = float(np.max(abs(reported_rate - gradient_rate)))
    _require((max(port_residual, derivative_residual, reported_residual) / relative_scale < 1e-10), "Numerical acceptance check failed")
    connection_rate = states[:, :2] - states[:, 2, None]
    velocity_rate = states[:, 3:5] - states[:, 5, None]
    mechanical_gradient = (gradient_rate + np.sum(model.masses * states[:, 3:6] * derivatives[:, 3:6], axis=1)
                           + np.sum(model.stiffness * connection_rate * velocity_rate, axis=1))
    mechanical_balance = (-np.sum(force_increment * (states[:, 3:5] - beam_velocity), axis=1)
                          - np.sum(model.damping * velocity_rate ** 2, axis=1) - dissipation + external)
    mechanical_residual = float(np.max(abs(mechanical_gradient - mechanical_balance)))
    mechanical_scale = max(float(np.max(abs(mechanical_balance))), relative_scale)
    _require((mechanical_residual / mechanical_scale < 1e-10), "Numerical acceptance check failed")
    reported_mechanical = np.array([port["mechanical_energy_rate"] for port in ports])
    reported_mechanical_residual = float(np.max(abs(reported_mechanical - mechanical_gradient)))
    _require((reported_mechanical_residual / mechanical_scale < 1e-10), "Numerical acceptance check failed")
    integrated = float(simpson(net_power, x=time))
    energy_change = float(energy[-1] - energy[0])
    integral_residual = energy_change - integrated
    energy_scale = max(float(np.max(energy)), abs(integrated), np.finfo(float).tiny)
    _require((abs(integral_residual) / energy_scale < 1e-3), "Numerical acceptance check failed")
    mask = (time >= fit_start) & (time <= fit_end)
    interval = time[mask][-1] - time[mask][0]
    return {"point_modal_power_max_difference_w": port_residual,
            "beam_derivative_balance_max_difference_w": derivative_residual,
            "reported_beam_derivative_max_difference_w": reported_residual,
            "beam_power_relative_residual": max(port_residual, derivative_residual, reported_residual) / relative_scale,
            "mechanical_balance_max_difference_w": mechanical_residual,
            "reported_mechanical_derivative_max_difference_w": reported_mechanical_residual,
            "mechanical_balance_relative_residual": max(mechanical_residual, reported_mechanical_residual) / mechanical_scale,
            "beam_energy_initial_j": float(energy[0]), "beam_energy_final_j": float(energy[-1]),
            "beam_energy_change_j": energy_change, "integrated_net_beam_power_j": integrated,
            "energy_integral_difference_j": integral_residual,
            "energy_integral_relative_difference": abs(integral_residual) / energy_scale,
            "magnetic_work_j": float(simpson(magnetic, x=time)),
            "external_work_j": float(simpson(external, x=time)),
            "beam_dissipated_energy_j": float(simpson(dissipation, x=time)),
            "fit_mean_magnetic_power_w": float(simpson(magnetic[mask], x=time[mask]) / interval),
            "fit_mean_beam_dissipation_w": float(simpson(dissipation[mask], x=time[mask]) / interval)}


def _rigid_time_check(cfg: dict, gain: float) -> dict:
    """Independent rigid nonlinear decay, retaining the eight active coordinates."""
    from .model import Model

    configuration = deepcopy(cfg)
    configuration["controller"]["current_gain_v_a"] = gain
    configuration["beam"]["enabled"] = False
    duration, dt = cfg["validation"]["duration_s"], cfg["time"]["dt"]
    configuration["time"] = {"dt": dt, "n_steps": int(round(duration / dt)) + 1}
    configuration["excitation"].update(type="none", amplitudes_n=[0.0, 0.0])
    configuration["initial"]["offsets"] = {}
    configuration["initial"]["random_std"] = {name: value for name, value in cfg["initial"]["random_std"].items()
                                               if name in ("z_m1", "z_m2", "z_c")}
    model = Model(configuration)
    initial = model.initial_state(np.random.default_rng(cfg["validation"]["seed"]))
    _require(np.max(abs(initial[model.q_slice])) == 0 and np.max(abs(initial[model.p_slice])) == 0)
    run = simulate(model, initial)
    _require(run["success"] and run["valid"].all(), run["solver_message"])
    indices = np.r_[np.arange(6), model.n_state - 2, model.n_state - 1]
    scales = model.scales[indices]
    increments = (run["X"][:, indices] - model.x_star[indices]) / scales
    norms = np.linalg.norm(increments, axis=1)
    matrix = model.rigid_matrix() * scales[None, :] / scales[:, None]
    values, vectors = np.linalg.eig(matrix)
    candidates = np.flatnonzero(values.imag > 0)
    dominant = candidates[np.argmax(values[candidates].real)]
    signal = increments @ np.linalg.inv(vectors)[dominant]
    # Rigid decay reaches floating-point equilibrium before the common 2--9.5s
    # fit interval; use an explicit early interval and report the entire decay.
    mask = (run["t"] >= 0.1) & (run["t"] <= 1.0)
    alpha = float(np.polyfit(run["t"][mask], np.log(abs(signal[mask])), 1)[0])
    frequency = float(abs(np.polyfit(run["t"][mask], np.unwrap(np.angle(signal[mask])), 1)[0]) / (2 * np.pi))
    _require(abs(alpha - values[dominant].real) < 0.05 * abs(values[dominant].real))
    _require(abs(frequency - values[dominant].imag / (2 * np.pi)) < 0.01 * frequency)
    late_peak = float(np.max(norms[run["t"] >= duration - 1.0]))
    decay_ratio = late_peak / float(norms[0])
    _require(decay_ratio < 1e-4)
    beam_peak = float(np.max(abs(run["X"][:, np.r_[np.arange(model.q_slice.start, model.q_slice.stop),
                                                        np.arange(model.p_slice.start, model.p_slice.stop)]])))
    _require(beam_peak == 0)
    return {"gain_v_a": float(gain), "config": configuration, "initial_state": initial.tolist(),
            "newton_matrix": "SciPy default numerical Jacobian",
            "active_state_names": [model.names[index] for index in indices],
            "rigid_eigenvalues": _spectrum(model.rigid_matrix()),
            "rigid_spectral_abscissa_per_s": float(np.max(values.real)),
            "projected_fit_interval_s": [0.1, 1.0],
            "projected_alpha_per_s": alpha, "projected_frequency_hz": frequency,
            "scaled_initial_norm": float(norms[0]), "scaled_final_norm": float(norms[-1]),
            "scaled_final_second_peak_norm": late_peak, "late_to_initial_norm_ratio": decay_ratio,
            "beam_state_max_absolute": beam_peak, "n_valid": int(run["valid"].sum()),
            "termination_reason": run["termination_reason"]}


def validate_scientific(cfg: dict) -> dict:
    """Re-run the specified rigid/one/two-mode stability and energy experiment.

    Numerical targets are measured from the current Model implementation. The
    report retains actual configurations, initial states, spectra, and errors.
    """
    from .model import Model

    validation = cfg["validation"]
    if len(set(validation["gains_v_a"])) < 2:
        raise ValueError("Scientific comparison requires at least two distinct gains: first growing, last decaying")
    configured_damping = cfg["beam"]["damping_ratio"]
    if isinstance(configured_damping, (list, tuple, np.ndarray)) and len(configured_damping) < 2:
        raise ValueError("Scientific one/two-mode comparison requires a scalar damping ratio or two modal ratios")
    duration, fit_start, fit_end = validation["duration_s"], validation["fit_start_s"], validation["fit_end_s"]
    dt = cfg["time"]["dt"]
    _require((0 < fit_start < fit_end <= duration), "Numerical acceptance check failed")
    steps = int(round(duration / dt)) + 1
    _require((abs((steps - 1) * dt - duration) < 1e-10), "Numerical acceptance check failed")
    cases, runs = [], {}
    for gain in validation["gains_v_a"]:
        for modes in (1, 2):
            configuration = deepcopy(cfg)
            configuration["time"] = {"dt": dt, "n_steps": steps}
            configuration["controller"]["current_gain_v_a"] = gain
            configuration["beam"].update(enabled=True, n_modes=modes)
            if isinstance(configured_damping, (list, tuple, np.ndarray)):
                configuration["beam"]["damping_ratio"] = list(configured_damping[:modes])
            configuration["excitation"].update(type="none", amplitudes_n=[0.0, 0.0])
            configuration["initial"]["offsets"] = {}
            configuration["initial"]["random_std"] = {key: value for key, value in cfg["initial"]["random_std"].items()
                                                       if key in ("z_m1", "z_m2", "z_c", "q1", "q2") and (modes == 2 or key != "q2")}
            model = Model(configuration)
            rigid = model.rigid_matrix()
            rigid_abscissa = float(np.max(np.linalg.eigvals(rigid).real))
            _require((rigid_abscissa < 0), "Numerical acceptance check failed")
            values = np.linalg.eigvals(model.linear_matrix())
            abscissa = float(np.max(values.real))
            if gain == validation["gains_v_a"][0]:
                _require((abscissa > 0 and np.any((values.real > 0) & (abs(values.imag) > 0))), "Numerical acceptance check failed")
            if gain == validation["gains_v_a"][-1]:
                _require((abscissa < 0), "Numerical acceptance check failed")
            initial = model.initial_state(np.random.default_rng(validation["seed"]))
            run = simulate(model, initial)
            _require((run["success"] and run["valid"].all()), run["solver_message"])
            fit = _mode_fit(model, run, fit_start, fit_end)
            power = _energy_check(model, run, fit_start, fit_end)
            case = {"gain_v_a": float(gain), "n_modes": modes, "config": configuration,
                    "initial_state": initial.tolist(), "rigid_eigenvalues": _spectrum(rigid),
                    "rigid_spectral_abscissa_per_s": rigid_abscissa,
                    "coupled_eigenvalues": _spectrum(model.linear_matrix()),
                    "coupled_spectral_abscissa_per_s": abscissa, "fit": fit, "energy": power,
                    "gap_min_m": float(np.min(run["O"][:, :2])), "gap_max_m": float(np.max(run["O"][:, :2]))}
            cases.append(case)
            runs[(gain, modes)] = (model, run, initial, case)
    convergence = []
    for gain in validation["gains_v_a"]:
        model, base, initial, case = runs[(gain, 2)]
        refined_cfg = deepcopy(model.cfg)
        refined_cfg["integrator"]["max_step"] /= 2
        refined_model = Model(refined_cfg)
        refined = simulate(refined_model, initial)
        _require((refined["success"] and refined["valid"].all()), "Numerical acceptance check failed")
        fit = _mode_fit(refined_model, refined, fit_start, fit_end)
        gap_difference = float(np.max(abs(base["O"][:, :2] - refined["O"][:, :2])))
        displacement_difference = float(np.max(abs(base["Y"] - refined["Y"])))
        alpha_difference = abs(case["fit"]["projected_alpha_per_s"] - fit["projected_alpha_per_s"])
        frequency_difference = abs(case["fit"]["projected_frequency_hz"] - fit["projected_frequency_hz"])
        _require((gap_difference < 1e-6), "Numerical acceptance check failed")
        _require((alpha_difference < max(0.01 * abs(fit["projected_alpha_per_s"]), 1e-3)), "Numerical acceptance check failed")
        _require((frequency_difference < 0.01 * fit["projected_frequency_hz"]), "Numerical acceptance check failed")
        convergence.append({"gain_v_a": float(gain), "base_max_step_s": model.cfg["integrator"]["max_step"],
                            "refined_config": refined_cfg, "refined_fit": fit,
                            "gap_max_difference_m": gap_difference, "displacement_max_difference_m": displacement_difference,
                            "alpha_absolute_difference_per_s": float(alpha_difference),
                            "frequency_absolute_difference_hz": float(frequency_difference)})
    channels = []
    reference_model = runs[(validation["gains_v_a"][0], 2)][0]
    for label, sign in (("symmetric", 1), ("antisymmetric", -1)):
        channel_cfg = deepcopy(reference_model.cfg)
        channel_cfg["time"]["n_steps"] = int(round(min(1.0, duration) / dt)) + 1
        model = Model(channel_cfg)
        initial = model.x_star.copy()
        amplitude = cfg["initial"]["random_std"]["z_m1"]
        _require((amplitude > 0), "Numerical acceptance check failed")
        initial[0], initial[1] = amplitude, sign * amplitude
        run = simulate(model, initial)
        _require((run["success"]), "Numerical acceptance check failed")
        modal_peaks = np.max(abs(run["X"][:, model.q_slice]), axis=0)
        _require((modal_peaks[0 if sign == 1 else 1] > 1e-12), "Numerical acceptance check failed")
        # Nonlinear differential motion may feed the common mode quadratically.
        if sign == 1:
            _require((modal_peaks[1] < max(1e-12, modal_peaks[0] * 1e-8)), "Numerical acceptance check failed")
        channels.append({"channel": label, "config": channel_cfg, "initial_state": initial.tolist(),
                         "modal_peak_displacement_m": modal_peaks.tolist(),
                         "gap_min_m": float(np.min(run["O"][:, :2])), "gap_max_m": float(np.max(run["O"][:, :2]))})
    rigid_cases = [_rigid_time_check(cfg, gain) for gain in validation["gains_v_a"]]
    return {"passed": True, "seed": validation["seed"], "duration_s": duration,
            "fit_interval_s": [fit_start, fit_end], "cases": cases, "step_halving": convergence,
            "symmetry_channels": channels, "rigid_nonlinear_cases": rigid_cases,
            "modal_truncation_convergence_claimed": False}
