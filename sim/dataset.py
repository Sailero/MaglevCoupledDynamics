"""Version 2 tensors, provenance, retained attempts, and named-column plots."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from sim.model import MODEL_VERSION, Model, validate_config, validate_model
from sim.solver import simulate, validate_scientific, validate_solver

Y_NAMES = ["z_c", "z_m1", "z_m2", "w1", "w2"]
U_NAMES = ["P1", "P2"]
O_NAMES = ["gap1", "gap2", "a_m1", "a_m2", "a_c", "i1", "i2", "beam_port_power"]
ROOT = Path(__file__).resolve().parents[1]
PARAMETER_SECTIONS = ("geometry", "vehicle", "beam", "magnet", "controller")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _numeric_parameters(cfg: dict) -> dict[str, float]:
    result = {}

    def flatten(value, key):
        if isinstance(value, dict):
            for child, item in sorted(value.items()):
                flatten(item, f"{key}.{child}" if key else child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                flatten(item, f"{key}.{index}")
        elif isinstance(value, (float, int)):
            result[key] = float(value)

    for section in PARAMETER_SECTIONS:
        flatten(cfg[section], section)
    return result


def _parameter_slot(cfg: dict, path: str):
    parts = path.split(".")
    if parts[0] not in PARAMETER_SECTIONS or path in {"beam.n_modes", "beam.enabled"}:
        raise ValueError(f"parameter_ranges cannot change {path}")
    obj = cfg
    try:
        for part in parts[:-1]:
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        key = int(parts[-1]) if isinstance(obj, list) else parts[-1]
        old = obj[key]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown parameter path: {path}") from exc
    if isinstance(old, bool) or not isinstance(old, (float, int)):
        raise ValueError(f"parameter range requires a scalar: {path}")
    return obj, key


def _set_parameter(cfg: dict, path: str, value: float) -> None:
    obj, key = _parameter_slot(cfg, path)
    obj[key] = int(value) if path == "magnet.turns" else float(value)


def _split_labels(count: int, fractions: list, seed: int) -> np.ndarray:
    """Largest remainder allocation; ties follow train, val, test order."""
    quotas = count * np.asarray(fractions, dtype=float)
    sizes = np.floor(quotas).astype(int)
    order = np.argsort(-(quotas - sizes), kind="stable")
    sizes[order[:count - int(sizes.sum())]] += 1
    labels = np.repeat(np.array(["train", "val", "test"]), sizes)
    return labels[np.random.default_rng(seed).permutation(count)]


def _source_identity() -> dict:
    digest = hashlib.sha256()
    for relative in ("make_data.py", "sim/model.py", "sim/solver.py", "sim/dataset.py"):
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                   cwd=ROOT, capture_output=True, text=True,
                                   check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", True
    return {"source_sha256": digest.hexdigest(), "git_commit": revision, "git_dirty": dirty}


def _assemble_dataset(cfg: dict, *, progress: bool = False) -> dict[str, np.ndarray]:
    validate_config(cfg)
    base = Model(cfg)
    settings = cfg["dataset"]
    n, nt = settings["n_samples"], cfg["time"]["n_steps"]
    dtype = np.dtype(settings["dtype"])
    ranges = settings["parameter_ranges"]
    n_families = min(settings["n_parameter_families"], n) if ranges else 1
    sample_seq, parameter_seq = np.random.SeedSequence(settings["seed"]).spawn(2)
    seeds = np.array([s.generate_state(1, dtype=np.uint64)[0] for s in sample_seq.spawn(n)], dtype=np.uint64)
    family_configs = []
    for sequence in parameter_seq.spawn(n_families):
        realized = copy.deepcopy(cfg)
        rng = np.random.default_rng(sequence)
        for path, bounds in sorted(ranges.items()):
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"range for {path} must be [minimum, maximum]")
            low, high = map(float, bounds)
            if not (np.isfinite([low, high]).all() and low <= high):
                raise ValueError(f"invalid parameter range: {path}")
            _parameter_slot(realized, path)
            if path == "magnet.turns":
                value = rng.integers(int(np.ceil(low)), int(np.floor(high)) + 1)
            else:
                value = rng.uniform(low, high)
            _set_parameter(realized, path, value)
        family_configs.append(realized)
    family_id = np.arange(n, dtype=np.int64) % n_families
    split_count = n if settings["split_by"] == "trajectory" else n_families
    labels = _split_labels(split_count, settings["split_fractions"], settings["split_seed"])
    split = labels if settings["split_by"] == "trajectory" else labels[family_id]
    keys = list(_numeric_parameters(cfg))
    arrays = {
        "format_version": np.array(2, dtype=np.int64),
        "model_version": np.array(MODEL_VERSION),
        "t": cfg["time"]["dt"] * np.arange(nt, dtype=float),
        "dt": np.array(cfg["time"]["dt"]),
        "Y": np.full((n, nt, 5), np.nan, dtype=dtype),
        "U": np.full((n, nt, 2), np.nan, dtype=dtype),
        "X": np.full((n, nt, base.n_state), np.nan, dtype=dtype),
        "O": np.full((n, nt, 8), np.nan, dtype=dtype),
        "valid": np.zeros((n, nt), dtype=bool),
        "split": np.asarray(split), "family_id": family_id, "seeds": seeds,
        "sample_index": np.arange(n, dtype=np.int64),
        "Y_names": np.array(Y_NAMES), "U_names": np.array(U_NAMES),
        "X_names": np.array(base.names), "O_names": np.array(O_NAMES),
        "Y_units": np.array(["m"] * 5), "U_units": np.array(["N"] * 2),
        "X_units": np.array(["m"] * 3 + ["m/s"] * 3 + ["m"] * base.n_modes + ["m/s"] * base.n_modes + ["A"] * 2),
        "O_units": np.array(["m", "m", "m/s2", "m/s2", "m/s2", "A", "A", "W"]),
        "parameter_names": np.array(keys), "parameter_values": np.full((n, len(keys)), np.nan),
        "initial_state": np.full((n, base.n_state), np.nan),
        "equilibrium_state": np.full((n, base.n_state), np.nan),
        "equilibrium_force": np.full((n, 2), np.nan),
        "equilibrium_current": np.full((n, 2), np.nan),
        "equilibrium_q": np.full((n, base.n_modes), np.nan),
        "equilibrium_w": np.full((n, 2), np.nan),
        "termination_time": np.full(n, np.nan),
        "termination_state": np.full((n, base.n_state), np.nan),
        "valid_length": np.zeros(n, dtype=np.int64), "success": np.zeros(n, dtype=bool),
        "config_json": np.array(_json(cfg)),
        "config_sha256": np.array(hashlib.sha256(_json(cfg).encode()).hexdigest()),
        "validation_json": np.array("{}"),
    }
    reasons, messages, actual_configs = [], [], []
    for index in range(n):
        realized = family_configs[family_id[index]]
        actual_configs.append(_json(realized))
        arrays["parameter_values"][index] = [_numeric_parameters(realized)[key] for key in keys]
        try:
            model = Model(realized)
        except ValueError as exc:
            reasons.append("invalid_parameters")
            messages.append(str(exc))
            if progress:
                print(f"[{index+1}/{n}] valid=0/{nt}, invalid_parameters: {exc}", flush=True)
            continue
        x0 = model.initial_state(np.random.default_rng(seeds[index]))
        arrays["initial_state"][index] = x0
        arrays["equilibrium_state"][index] = model.x_star
        arrays["equilibrium_force"][index] = model.force_star
        arrays["equilibrium_current"][index] = model.i0
        arrays["equilibrium_q"][index] = model.q_star
        arrays["equilibrium_w"][index] = model.w_star
        result = simulate(model, x0)
        for key in ("X", "Y", "U", "O", "valid", "termination_time", "termination_state", "success"):
            arrays[key][index] = result[key]
        arrays["valid_length"][index] = np.count_nonzero(result["valid"])
        reasons.append(result["termination_reason"])
        messages.append(result["solver_message"])
        if progress:
            print(f"[{index+1}/{n}] valid={arrays['valid_length'][index]}/{nt}, {reasons[-1]}", flush=True)
    arrays["termination_reason"] = np.array(reasons)
    arrays["solver_message"] = np.array(messages)
    arrays["actual_config_json"] = np.array(actual_configs)
    for key, value in _source_identity().items():
        arrays[key] = np.array(value)
    return arrays


def _save(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Read without pickle and enforce the complete persisted v2 contract."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    try:
        _check_dataset(data)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError(f"malformed format v2 dataset: {exc}") from exc
    return data


def _check_dataset(data: dict[str, np.ndarray]) -> None:
    required = {
        "format_version", "model_version", "t", "dt", "Y", "U", "X", "O",
        "valid", "valid_length", "sample_index", "seeds", "split", "family_id",
        "parameter_names", "parameter_values", "config_json", "actual_config_json",
        "initial_state", "equilibrium_state", "equilibrium_force", "equilibrium_current",
        "equilibrium_q", "equilibrium_w", "success", "termination_reason",
        "termination_time", "termination_state", "solver_message", "config_sha256",
        "source_sha256", "git_commit", "git_dirty", "validation_json",
    } | {f"{key}_{suffix}" for key in ("X", "Y", "U", "O") for suffix in ("names", "units")}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"missing format v2 fields: {sorted(missing)}")
    if any(value.dtype.kind == "O" for value in data.values()):
        raise ValueError("object arrays are not part of format v2")

    def field(key, shape, kinds):
        value = data[key]
        if value.shape != shape or value.dtype.kind not in kinds:
            raise ValueError(f"invalid {key} shape or dtype")
        return value

    field("format_version", (), "iu")
    field("git_dirty", (), "b")
    field("dt", (), "f")
    for key in ("model_version", "config_json", "config_sha256", "source_sha256",
                "git_commit", "validation_json"):
        field(key, (), "U")
    if data["format_version"].item() != 2 or str(data["model_version"]) != MODEL_VERSION:
        raise ValueError("expected stationary two-point format v2")

    def json_object(text, name):
        value = json.loads(str(text))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")
        _json(value)  # Also rejects Python's permissively parsed NaN/Infinity.
        return value

    cfg = json_object(data["config_json"], "config_json")
    validate_config(cfg)
    json_object(data["validation_json"], "validation_json")

    def hex_string(value, lengths):
        return len(value) in lengths and all(char in "0123456789abcdefABCDEF" for char in value)

    for key in ("config_sha256", "source_sha256"):
        if not hex_string(str(data[key]), {64}):
            raise ValueError(f"invalid {key}")
    revision = str(data["git_commit"])
    if revision != "unavailable" and not hex_string(revision, {40, 64}):
        raise ValueError("invalid git_commit")
    if hashlib.sha256(_json(cfg).encode()).hexdigest() != str(data["config_sha256"]):
        raise ValueError("configuration checksum mismatch")

    n, nt = cfg["dataset"]["n_samples"], cfg["time"]["n_steps"]
    j = cfg["beam"]["n_modes"]
    widths = {"Y": 5, "U": 2, "X": 8 + 2 * j, "O": 8}
    expected_names = {"Y": Y_NAMES, "U": U_NAMES, "O": O_NAMES}
    expected_names["X"] = ["z_m1", "z_m2", "z_c", "v_m1", "v_m2", "v_c"] + [f"q{k}" for k in range(1, j+1)] + [f"p{k}" for k in range(1, j+1)] + ["i1", "i2"]
    expected_units = {"Y": ["m"] * 5, "U": ["N"] * 2,
                      "X": ["m"] * 3 + ["m/s"] * 3 + ["m"] * j + ["m/s"] * j + ["A"] * 2,
                      "O": ["m", "m", "m/s2", "m/s2", "m/s2", "A", "A", "W"]}
    field("valid", (n, nt), "b")
    if np.any(np.diff(data["valid"].astype(int), axis=1) > 0):
        raise ValueError("valid samples must form a prefix")
    for key, width in widths.items():
        field(key, (n, nt, width), "f")
        if data[key].dtype != np.dtype(cfg["dataset"]["dtype"]):
            raise ValueError(f"{key} precision disagrees with configuration")
        field(f"{key}_names", (width,), "U")
        field(f"{key}_units", (width,), "U")
        if data[f"{key}_names"].tolist() != expected_names[key] or data[f"{key}_units"].tolist() != expected_units[key]:
            raise ValueError(f"invalid {key} column names or units")
        if not np.isfinite(data[key][data["valid"]]).all():
            raise ValueError(f"nonfinite valid {key} values")
        if not np.isnan(data[key][~data["valid"]]).all():
            raise ValueError(f"invalid {key} tail must be NaN")
    field("t", (nt,), "f")
    dt = float(data["dt"])
    if not np.isfinite(dt) or dt <= 0 or dt != cfg["time"]["dt"] or not np.array_equal(data["t"], dt * np.arange(nt)):
        raise ValueError("sample grid must be dt*arange(T)")
    for key in ("valid_length", "sample_index", "family_id"):
        field(key, (n,), "iu")
    if not np.array_equal(data["valid_length"], data["valid"].sum(axis=1)):
        raise ValueError("valid lengths disagree with mask")
    if not np.array_equal(data["sample_index"], np.arange(n)):
        raise ValueError("sample_index must preserve the planned arange(N)")
    if np.any(data["family_id"] < 0):
        raise ValueError("family_id must be nonnegative")
    field("seeds", (n,), "u")
    if data["seeds"].dtype != np.dtype(np.uint64):
        raise ValueError("seeds must have uint64 dtype")
    field("success", (n,), "b")
    for key in ("split", "actual_config_json", "termination_reason", "solver_message"):
        field(key, (n,), "U")
    field("termination_time", (n,), "f")
    for key, width in (("termination_state", widths["X"]), ("initial_state", widths["X"]),
                       ("equilibrium_state", widths["X"]), ("equilibrium_force", 2),
                       ("equilibrium_current", 2), ("equilibrium_q", j), ("equilibrium_w", 2)):
        field(key, (n, width), "f")
    parameters = list(_numeric_parameters(cfg))
    field("parameter_names", (len(parameters),), "U")
    if data["parameter_names"].tolist() != parameters:
        raise ValueError("physical parameter names disagree with configuration")
    field("parameter_values", (n, len(parameters)), "f")
    if not np.isfinite(data["parameter_values"]).all():
        raise ValueError("physical parameter rows must be finite")
    if set(data["split"].tolist()) - {"train", "val", "test"}:
        raise ValueError("invalid split labels")

    for family in np.unique(data["family_id"]):
        members = data["family_id"] == family
        rows = data["parameter_values"][members]
        if not np.all(rows == rows[0]):
            raise ValueError("a parameter family has inconsistent physical parameters")
        if cfg["dataset"]["split_by"] == "family" and len(np.unique(data["split"][members])) != 1:
            raise ValueError("a parameter family crosses dataset splits")

    def same_structure(reference, actual):
        if isinstance(reference, dict):
            return isinstance(actual, dict) and reference.keys() == actual.keys() and all(same_structure(reference[key], actual[key]) for key in reference)
        if isinstance(reference, list):
            return isinstance(actual, list) and len(reference) == len(actual) and all(same_structure(left, right) for left, right in zip(reference, actual))
        if isinstance(reference, bool):
            return isinstance(actual, bool)
        if isinstance(reference, (int, float)):
            return isinstance(actual, (int, float)) and not isinstance(actual, bool)
        return isinstance(actual, type(reference))

    physical_reasons = {f"{kind}{point}_{side}" for kind in ("gap", "current") for point in (1, 2) for side in ("min", "max")}
    allowed_reasons = physical_reasons | {"completed", "solver_failure", "invalid_parameters"}
    if set(data["termination_reason"].tolist()) - allowed_reasons:
        raise ValueError("unknown termination reason")
    if not np.array_equal(data["success"], data["termination_reason"] == "completed"):
        raise ValueError("success must agree with completed termination")
    time_tolerance = 32 * np.finfo(float).eps * max(1.0, float(data["t"][-1]))
    state_rtol = 8 * np.finfo(data["X"].dtype).eps

    for index, text in enumerate(data["actual_config_json"]):
        actual = json_object(text, f"actual_config_json[{index}]")
        if not same_structure(cfg, actual) or any(actual[section] != cfg[section] for section in cfg if section not in PARAMETER_SECTIONS):
            raise ValueError("actual configuration structure or run settings disagree")
        if not isinstance(actual["beam"]["n_modes"], int) or actual["beam"]["n_modes"] != j or actual["beam"]["enabled"] != cfg["beam"]["enabled"] or not isinstance(actual["magnet"]["turns"], int):
            raise ValueError("actual configuration changes modal order, beam enablement, or integer turns")
        row = _numeric_parameters(actual)
        if list(row) != parameters or not np.array_equal(data["parameter_values"][index], [row[key] for key in parameters]):
            raise ValueError("actual parameters disagree with parameter row")
        length = int(data["valid_length"][index])
        reason, time = str(data["termination_reason"][index]), float(data["termination_time"][index])
        state, initial = data["termination_state"][index], data["initial_state"][index]
        if reason == "invalid_parameters":
            if length or np.isinf(time) or (np.isfinite(time) and not 0 <= time <= data["t"][-1]):
                raise ValueError("invalid parameters require no valid samples and an undefined or in-range termination time")
            if any(np.isinf(data[key][index]).any() for key in ("initial_state", "termination_state", "equilibrium_state", "equilibrium_force", "equilibrium_current", "equilibrium_q", "equilibrium_w")):
                raise ValueError("undefined parameter diagnostics may be NaN, never infinity")
            continue
        if not np.isfinite(time) or not 0 <= time <= data["t"][-1]:
            raise ValueError("termination time must be finite and within the requested grid")
        if not length and time != 0:
            raise ValueError("a failure without an initial valid sample must terminate at zero")
        if length and time < data["t"][length-1] - time_tolerance:
            raise ValueError("termination precedes a valid sample")
        for key in ("equilibrium_state", "equilibrium_force", "equilibrium_current", "equilibrium_q", "equilibrium_w"):
            if not np.isfinite(data[key][index]).all():
                raise ValueError(f"nonfinite {key} for a constructed model")
        initial_failure = (reason == "solver_failure" and length == 0 and time == 0
                           and np.array_equal(state, initial, equal_nan=True))
        if not np.isfinite(state).all() and not initial_failure:
            raise ValueError("nonfinite termination state is only valid for a retained time-zero initial failure")
        if reason != "solver_failure" and (not np.isfinite(state).all() or not np.isfinite(initial).all()):
            raise ValueError("physical terminations require finite initial and complete terminal states")
        if length and (not np.isfinite(initial).all() or not np.allclose(initial, data["X"][index, 0], rtol=state_rtol, atol=1e-12)):
            raise ValueError("initial state disagrees with the first valid sample")
        if time == 0 and not np.allclose(state, initial, rtol=state_rtol, atol=1e-12, equal_nan=True):
            raise ValueError("time-zero termination must retain the initial state")
        if reason == "completed":
            if length != nt or abs(time - data["t"][-1]) > time_tolerance or not np.allclose(state, data["X"][index, -1], rtol=state_rtol, atol=1e-12):
                raise ValueError("completed trajectory must retain every sample and its final state")
        if reason in physical_reasons:
            if length < nt and data["t"][length] <= time + time_tolerance and not (length == 0 and time == 0):
                raise ValueError("physical event omits a sample before termination")
            magnet, beam = actual["magnet"], actual["beam"]
            phi = np.sin(np.pi * np.asarray(actual["geometry"]["positions_m"])[:, None] / beam["length_m"] * np.arange(1, j + 1)) if beam["enabled"] else np.zeros((2, j))
            values = {"gap": magnet["gap_m"] + state[:2] - phi @ state[6:6+j], "current": state[-2:]}
            kind, side = reason.rsplit("_", 1)
            point, kind = int(kind[-1]) - 1, kind[:-1]
            boundary = magnet[f"{kind}_{side}_{'m' if kind == 'gap' else 'a'}"]
            distance = (values[kind][point] - boundary) * (1 if side == "min" else -1)
            boundary_tolerance = 1e-10 * max(1.0, abs(boundary))
            outside_tolerance = 32 * np.finfo(float).eps * max(1.0, abs(distance))
            if (length == 0 and time == 0 and distance >= -outside_tolerance) or ((length or time > 0) and abs(distance) > boundary_tolerance):
                raise ValueError("physical termination state disagrees with its named boundary")


def generate_dataset(cfg: dict, output_dir: str | Path = ROOT / "data") -> Path:
    """Run each planned sample exactly once, validate, and atomically save."""
    validate_config(cfg)
    reports = {}
    for flag, check in (("model", validate_model), ("solver", validate_solver),
                        ("dataset", validate_dataset), ("scientific", validate_scientific)):
        if cfg["validation"][flag]:
            print(f"Validating {flag} ...", flush=True)
            reports[flag] = check(cfg)
    arrays = _assemble_dataset(cfg, progress=True)
    arrays["validation_json"] = np.array(_json(reports))
    path = Path(output_dir) / f"{cfg['dataset']['name']}.npz"
    _save(path, arrays)
    data = load_dataset(path)
    if reports:
        report_path = path.with_name(path.stem + "_validation.json")
        report_path.write_text(_json(reports) + "\n", encoding="utf-8")
    print(f"Saved {path}\nY={data['Y'].shape}, U={data['U'].shape}, X={data['X'].shape}, O={data['O'].shape}", flush=True)
    if cfg["dataset"]["plot"]:
        plot_dataset(path)
    return path


def plot_dataset(path: str | Path) -> Path:
    """Use persisted column names; never infer physical meaning by index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    data = load_dataset(path)
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    groups = [("Y", ["z_c", "z_m1", "z_m2"], "Displacement (m)"),
              ("Y", ["w1", "w2"], "Beam (m)"),
              ("O", ["gap1", "gap2"], "Gap (m)"),
              ("O", ["i1", "i2"], "Current (A)"),
              ("O", ["beam_port_power"], "Beam input (W)")]
    for ax, (key, columns, label) in zip(axes, groups):
        names = data[f"{key}_names"].tolist()
        for sample in range(min(3, len(data[key]))):
            valid = data["valid"][sample]
            for column in columns:
                ax.plot(data["t"][valid], data[key][sample, valid, names.index(column)],
                        label=f"{sample}: {column}", linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(loc="upper right", ncol=min(3, len(ax.lines)), fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{path.stem} | format v2 | valid prefixes")
    target = path.with_suffix(".png")
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target


def validate_dataset(cfg: dict) -> dict:
    """Small real integrations verify roundtrip, replay, failures, families."""
    small = copy.deepcopy(cfg)
    small["dataset"].update(n_samples=4, n_parameter_families=2, plot=False,
                            parameter_ranges={"controller.current_gain_v_a": [65.0, 80.0]},
                            split_by="family")
    small["time"]["n_steps"] = 31
    small["initial"]["offsets"] = {}
    small["initial"]["random_std"] = {}
    small["excitation"]["type"] = "none"
    small["excitation"]["amplitudes_n"] = [0.0, 0.0]
    first, repeat = _assemble_dataset(small), _assemble_dataset(small)
    for key in ("X", "Y", "U", "O", "parameter_values", "split", "seeds", "initial_state"):
        identical = (np.array_equal(first[key], repeat[key], equal_nan=True)
                     if first[key].dtype.kind == "f" else np.array_equal(first[key], repeat[key]))
        if not identical:
            raise AssertionError(f"replay mismatch: {key}")
    failed = copy.deepcopy(small)
    failed["initial"]["offsets"] = {"z_m1": -2 * cfg["magnet"]["gap_m"]}
    attempts = _assemble_dataset(failed)
    if attempts["valid"].any() or not np.isnan(attempts["X"]).all() or len(attempts["seeds"]) != 4:
        raise AssertionError("failed attempts must retain every index and NaN tail")
    if not np.isfinite(attempts["termination_state"]).all() or not np.all(attempts["termination_time"] == 0):
        raise AssertionError("initially invalid trajectories need exact termination states")
    with tempfile.TemporaryDirectory() as directory:
        for index, arrays in enumerate((first, attempts)):
            path = Path(directory) / f"roundtrip{index}.npz"
            _save(path, arrays)
            load_dataset(path)
        fixture_cfg = copy.deepcopy(small)
        fixture_cfg["time"]["n_steps"] = 2
        fixture_cfg["dataset"].update(n_samples=2, n_parameter_families=1)
        fixture = _assemble_dataset(fixture_cfg)
        fixture["validation_json"] = np.array("{}")
        rejected = []

        def reject(name, mutate, source=fixture):
            damaged = copy.deepcopy(source)
            mutate(damaged)
            path = Path(directory) / "damaged.npz"
            _save(path, damaged)
            try:
                load_dataset(path)
            except ValueError:
                rejected.append(name)
                return
            raise AssertionError(f"reader accepted malformed dataset: {name}")

        for key in fixture:
            reject(f"missing {key}", lambda data, key=key: data.pop(key))
        for key in ("format_version", "model_version", "dt", "config_json", "config_sha256",
                    "source_sha256", "git_commit", "git_dirty", "validation_json"):
            reject(f"nonscalar {key}", lambda data, key=key: data.update({key: data[key][None]}))
        for key in ("config_json", "validation_json"):
            reject(f"malformed {key}", lambda data, key=key: data.update({key: np.array("{")}))
        reject("malformed actual config", lambda data: data.update(actual_config_json=np.array(["{", "{}"])))
        reject("nonfinite JSON", lambda data: data.update(validation_json=np.array('{"bad": NaN}')))
        reject("bad source hash", lambda data: data.update(source_sha256=np.array("bad")))
        reject("bad git revision", lambda data: data.update(git_commit=np.array("bad")))
        reject("bad configuration hash", lambda data: data.update(config_sha256=np.array("0" * 64)))
        reject("float seeds", lambda data: data.update(seeds=data["seeds"].astype(float)))
        reject("duplicate indices", lambda data: data.update(sample_index=np.array([1, 1])))
        reject("negative family", lambda data: data.update(family_id=np.array([-1, -1])))
        reject("float family", lambda data: data.update(family_id=data["family_id"].astype(float)))
        reject("float valid length", lambda data: data.update(valid_length=data["valid_length"].astype(float)))
        reject("integer success", lambda data: data.update(success=data["success"].astype(int)))
        reject("integer git dirty", lambda data: data.update(git_dirty=np.array(1)))
        reject("nonfinite termination", lambda data: data["termination_time"].fill(np.nan))
        reject("late termination", lambda data: data["termination_time"].fill(999.0))
        reject("early completion", lambda data: data["termination_time"].fill(0.0))
        reject("unknown reason", lambda data: data.update(termination_reason=np.array(["invented"] * 2)))
        reject("completion without success", lambda data: data["success"].fill(False))
        reject("success with failure", lambda data: data.update(termination_reason=np.array(["solver_failure"] * 2)))
        reject("invalid parameters with samples", lambda data: (data["success"].fill(False), data.update(termination_reason=np.array(["invalid_parameters"] * 2))))
        reject("nonfinite physical state", lambda data: data["termination_state"].fill(np.nan))
        reject("nonfinite equilibrium", lambda data: data["equilibrium_state"].fill(np.nan))
        reject("wrong initial state shape", lambda data: data.update(initial_state=data["initial_state"][:, :-1]))
        reject("wrong tensor precision", lambda data: data.update(Y=data["Y"].astype(np.float32 if data["Y"].dtype == np.float64 else np.float64)))
        for key in ("X", "Y", "U", "O"):
            reject(f"wrong {key} units", lambda data, key=key: data.update({f"{key}_units": np.array(["bad"] * len(data[f"{key}_units"]))}))

        def alter_actual(data, section, field, value, *, update_row=False):
            actual = json.loads(str(data["actual_config_json"][1]))
            actual[section][field] = value
            texts = data["actual_config_json"].tolist()
            texts[1] = _json(actual)
            data["actual_config_json"] = np.array(texts)
            if update_row:
                column = data["parameter_names"].tolist().index(f"{section}.{field}")
                data["parameter_values"][1, column] = value

        reject("parameter row mismatch", lambda data: alter_actual(data, "vehicle", "m1_kg", 301.0))
        reject("inconsistent family", lambda data: alter_actual(data, "vehicle", "m1_kg", 301.0, update_row=True))
        reject("actual modal order", lambda data: alter_actual(data, "beam", "n_modes", 1, update_row=True))
        reject("actual time grid", lambda data: alter_actual(data, "time", "dt", 0.002))
        reject("actual model version", lambda data: alter_actual(data, "meta", "model_version", "other"))

        def no_valid_failure(data, reason, time):
            data["success"][:] = False
            data["termination_reason"] = np.array([reason] * 2)
            data["termination_time"][:] = time
            data["valid"][:] = False
            data["valid_length"][:] = 0
            for key in ("X", "Y", "U", "O"):
                data[key][:] = np.nan

        reject("late failure without initial sample", lambda data: no_valid_failure(data, "solver_failure", data["t"][-1]))
        reject("infinite undefined initial state", lambda data: (no_valid_failure(data, "invalid_parameters", np.nan), data["initial_state"].fill(np.inf)))
        reject("terminal state disagrees with final sample", lambda data: data["termination_state"].__setitem__((0, 0), 1.0))

        def nonfinite_accepted_failure(data, value):
            data["success"][:] = False
            data["termination_reason"] = np.array(["solver_failure"] * 2)
            data["termination_time"][:] = data["t"][-1] / 2
            data["valid"][:, 1:] = False
            data["valid_length"][:] = 1
            for key in ("X", "Y", "U", "O"):
                data[key][:, 1:] = np.nan
            data["termination_state"][:] = value

        for name, value in (("NaN", np.nan), ("Inf", np.inf)):
            reject(f"nonfinite accepted solver-failure state {name}", lambda data, value=value: nonfinite_accepted_failure(data, value))

        # Real solver records retain nonfinite initial diagnostics at time zero.
        diagnostics = copy.deepcopy(fixture)
        diagnostics["termination_reason"] = np.array(["solver_failure"] * 2)
        for index, text in enumerate(diagnostics["actual_config_json"]):
            model = Model(json.loads(str(text)))
            initial = model.x_star.copy()
            initial[0] = np.nan if index == 0 else np.inf
            result = simulate(model, initial)
            diagnostics["initial_state"][index] = initial
            for key in ("X", "Y", "U", "O", "valid", "termination_time", "termination_state", "success"):
                diagnostics[key][index] = result[key]
            if result["termination_reason"] != "solver_failure":
                raise AssertionError("nonfinite initial state did not retain a solver failure")
            diagnostics["valid_length"][index] = 0
        _save(Path(directory) / "diagnostic.npz", diagnostics)
        load_dataset(Path(directory) / "diagnostic.npz")

        # Every physical reason accepts both an exact boundary and an overshoot.
        boundary_cfg = copy.deepcopy(fixture_cfg)
        boundary_cfg["dataset"]["parameter_ranges"] = {}
        boundary_model = Model(boundary_cfg)
        for kind in ("gap", "current"):
            for point in (1, 2):
                for side, direction in (("min", -1), ("max", 1)):
                    key = f"{kind}_{side}_{'m' if kind == 'gap' else 'a'}"
                    baseline = boundary_cfg["magnet"]["gap_m"] if kind == "gap" else boundary_model.i0[point - 1]
                    coordinate = f"z_m{point}" if kind == "gap" else f"i{point}"
                    for overshoot in (0.0, 1e-5):
                        boundary_cfg["initial"]["offsets"] = {coordinate: boundary_cfg["magnet"][key] - baseline + direction * overshoot}
                        boundary = _assemble_dataset(boundary_cfg)
                        expected_reason = f"{kind}{point}_{side}"
                        if not np.all(boundary["termination_reason"] == expected_reason) or not np.all(boundary["valid_length"] == (1 if overshoot == 0 else 0)):
                            raise AssertionError(f"unexpected boundary fixture: {expected_reason}")
                        _save(Path(directory) / "boundary.npz", boundary)
                        load_dataset(Path(directory) / "boundary.npz")
                        if overshoot == 0:
                            reject(f"missing exact-boundary sample {expected_reason}", lambda data: no_valid_failure(data, expected_reason, 0.0), source=boundary)

        # An integer YAML spelling must not turn a continuous mass into a lattice.
        fixture_cfg["dataset"]["parameter_ranges"] = {"vehicle.m1_kg": [299.1, 299.9]}
        fixture_cfg["vehicle"]["m1_kg"] = 300
        integer_spelling = _assemble_dataset(fixture_cfg)
        fixture_cfg["vehicle"]["m1_kg"] = 300.0
        float_spelling = _assemble_dataset(fixture_cfg)
        for key in ("parameter_values", "initial_state", "X", "Y", "U", "O"):
            if not np.array_equal(integer_spelling[key], float_spelling[key], equal_nan=True):
                raise AssertionError(f"continuous numeric spelling changes {key}")
    return {"passed": True, "repeat_exact": True, "allow_pickle": False,
            "retained_failed_attempts": 4, "family_split_checked": True,
            "malformed_datasets_rejected": len(rejected), "continuous_numeric_spelling": True,
            "nonfinite_initial_diagnostics": True, "physical_boundary_roundtrips": 16,
            "sample_shape": list(first["Y"].shape)}
