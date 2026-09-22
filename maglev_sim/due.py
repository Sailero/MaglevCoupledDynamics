"""生成 DUE 可直接加载的张量：Y (N,T,3) 与 U (N,T,2)。"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

from maglev_sim.plant.config import Config, demo_config
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.model import initial_state
from maglev_sim.plant.simulate import simulate

# 预测目标：车体、电磁铁、梁在悬浮点位移。已知输入：不平顺 r、梁上外力 P。
Y_NAMES = ("z_car", "z_magnet", "beam_displacement_at_magnet")
U_NAMES = ("irregularity", "force_P")

NPZ_KEYS = ("Y", "U", "t", "dt", "split", "Y_names", "U_names")


def estimate_due_dataset(
    n_samples: int,
    n_times: int,
    *,
    dt: float = 0.001,
    dtype: str | np.dtype = "float64",
    **_: Any,
) -> dict[str, Any]:
    """不仿真，只估算形状和内存。"""
    n_y, n_u = len(Y_NAMES), len(U_NAMES)
    itemsize = int(np.dtype(dtype).itemsize)
    bytes_y = n_samples * n_times * n_y * itemsize
    bytes_u = n_samples * n_times * n_u * itemsize
    t_end = float(dt) * (n_times - 1)
    return {
        "n_samples": n_samples,
        "n_times": n_times,
        "n_y": n_y,
        "n_u": n_u,
        "Y_names": list(Y_NAMES),
        "U_names": list(U_NAMES),
        "Y_shape": (n_samples, n_times, n_y),
        "U_shape": (n_samples, n_times, n_u),
        "dt": float(dt),
        "t_end": t_end,
        "dtype": str(np.dtype(dtype)),
        "bytes_total": bytes_y + bytes_u,
    }


def load_due_dataset(path: str | Path) -> dict[str, Any]:
    """读取生成器写出的 NPZ，只返回 DUE 字段。"""
    raw = np.load(path, allow_pickle=True)
    extra = set(raw.files) - set(NPZ_KEYS)
    if extra:
        raise ValueError(f"npz 含有非 DUE 字段: {sorted(extra)}")
    missing = set(NPZ_KEYS) - set(raw.files)
    if missing:
        raise ValueError(f"npz 缺少字段: {sorted(missing)}")
    return {
        "Y": np.asarray(raw["Y"]),
        "U": np.asarray(raw["U"]),
        "t": np.asarray(raw["t"]),
        "dt": float(raw["dt"]),
        "split": np.asarray(raw["split"]),
        "Y_names": tuple(str(x) for x in raw["Y_names"]),
        "U_names": tuple(str(x) for x in raw["U_names"]),
    }


def generate_due_dataset(
    n_samples: int,
    n_times: int,
    output_path: str | Path,
    *,
    dt: float = 0.001,
    config: Config | None = None,
    seed: int = 0,
    z_m_std: float = 2.0e-4,
    r_std: float = 0.0,
    dtype: str | np.dtype = "float64",
    max_attempts_factor: int = 20,
    verbose: bool = True,
) -> Path:
    """独立仿真 n_samples 条轨迹，写出 DUE 张量。

    NPZ 仅含：
      Y      (N, T, 3)  预测目标
      U      (N, T, 2)  已知输入（可含未来）
      t      (T,)
      dt
      split  (N,)        按整条轨迹划分 train/val/test
      Y_names, U_names
    """
    if n_samples < 1 or n_times < 2:
        raise ValueError("需要 n_samples >= 1 且 n_times >= 2")
    if dt <= 0.0:
        raise ValueError("dt 必须为正")

    cfg = copy.deepcopy(config) if config is not None else demo_config()
    cfg.output.dt = float(dt)
    cfg.output.t_start = 0.0
    t_grid = cfg.output.t_start + float(dt) * np.arange(n_times, dtype=float)
    cfg.output.t_end = float(t_grid[-1])
    cfg.validate()

    np_dtype = np.dtype(dtype)
    info = estimate_due_dataset(n_samples, n_times, dt=dt, dtype=np_dtype)
    if verbose:
        print(
            f"将生成 Y{info['Y_shape']} + U{info['U_shape']}，"
            f"dtype={np_dtype}，约 {info['bytes_total'] / 1e6:.2f} MB，"
            f"t_end={info['t_end']:.6g} s",
            file=sys.stderr,
            flush=True,
        )

    Y = np.empty((n_samples, n_times, len(Y_NAMES)), dtype=np_dtype)
    U = np.empty((n_samples, n_times, len(U_NAMES)), dtype=np_dtype)

    accepted = 0
    attempt = 0
    limit = n_samples * max_attempts_factor
    while accepted < n_samples:
        attempt += 1
        if attempt > limit:
            raise RuntimeError(
                f"尝试 {attempt - 1} 次后只得到 {accepted}/{n_samples} 条完整轨迹"
            )
        sample_seed = int(seed) + attempt
        run_cfg = copy.deepcopy(cfg)
        rng = np.random.default_rng(sample_seed)
        extra_std = dict(run_cfg.initial.random_std)
        if z_m_std:
            extra_std["z_m"] = float(z_m_std)
        run_cfg.initial.random_std = extra_std
        if r_std:
            run_cfg.irregularity.type = "constant"
            run_cfg.irregularity.value = float(run_cfg.irregularity.value) + float(
                r_std * rng.standard_normal()
            )
        y0 = initial_state(run_cfg, seed=sample_seed)
        exc = Excitation.from_config(run_cfg, seed=sample_seed)
        traj = simulate(run_cfg, y0, exc, t_grid)
        if (not traj.completed) or traj.t.size != n_times:
            if verbose:
                print(
                    f"丢弃 seed={sample_seed} reason={traj.termination_reason} n={traj.t.size}",
                    file=sys.stderr,
                    flush=True,
                )
            continue
        for k, name in enumerate(Y_NAMES):
            Y[accepted, :, k] = np.asarray(getattr(traj, name), dtype=np_dtype)
        for k, name in enumerate(U_NAMES):
            U[accepted, :, k] = np.asarray(getattr(traj, name), dtype=np_dtype)
        accepted += 1
        if verbose and (accepted == 1 or accepted % 10 == 0 or accepted == n_samples):
            print(f"已完成 {accepted}/{n_samples}", file=sys.stderr, flush=True)

    n_train = min(int(round(0.70 * n_samples)), n_samples)
    n_val = min(int(round(0.15 * n_samples)), n_samples - n_train)
    split = np.array(
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n_samples - n_train - n_val),
        dtype=object,
    )
    split = split[np.random.default_rng(int(seed)).permutation(n_samples)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        Y=Y,
        U=U,
        t=np.asarray(t_grid, dtype=np_dtype),
        dt=np.float64(dt),
        split=split,
        Y_names=np.asarray(Y_NAMES),
        U_names=np.asarray(U_NAMES),
    )
    if verbose:
        print(f"已写入 {output_path}", file=sys.stderr, flush=True)
    return output_path
