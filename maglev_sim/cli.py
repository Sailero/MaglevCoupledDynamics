"""命令行：生成 DUE 张量 Y (N,T,3) 与 U (N,T,2)。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maglev_sim.due import estimate_due_dataset, generate_due_dataset
from maglev_sim.plant.config import load_config
from maglev_sim.plot import plot_due_dataset


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成 DUE 训练张量：Y (样本, 时间, 3)，U (样本, 时间, 2)。"
    )
    parser.add_argument("--n-samples", type=int, required=True, help="样本条数 N")
    parser.add_argument("--n-times", type=int, required=True, help="每条时间点数 T")
    parser.add_argument("--dt", type=float, default=0.001, help="采样周期 [s]")
    parser.add_argument("--out", type=Path, default=Path("data/due_dataset.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--z-m-std", type=float, default=2.0e-4, help="电磁铁初值扰动 [m]")
    parser.add_argument("--r-std", type=float, default=2.0e-4, help="每条样本常值不平顺标准差 [m]")
    parser.add_argument("--plot", action="store_true", help="写出 Y/U 矩阵图和样本曲线")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else None
    info = estimate_due_dataset(
        args.n_samples, args.n_times, dt=args.dt, dtype=args.dtype
    )
    print(f"Y {info['Y_shape']}  <- {info['Y_names']}")
    print(f"U {info['U_shape']}  <- {info['U_names']}")
    print(f"dt = {info['dt']} s，t_end = {info['t_end']:.6g} s，dtype = {info['dtype']}")
    print(f"内存 ≈ {info['bytes_total'] / 1e6:.2f} MB")
    if args.dry_run:
        print("dry-run：未仿真，未写文件。")
        return 0
    path = generate_due_dataset(
        args.n_samples,
        args.n_times,
        args.out,
        dt=args.dt,
        config=cfg,
        seed=args.seed,
        z_m_std=args.z_m_std,
        r_std=args.r_std,
        dtype=args.dtype,
        verbose=not args.quiet,
    )
    print(f"wrote {path}")
    if args.plot:
        figures = plot_due_dataset(path)
        for fig in figures:
            print(f"plot {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
