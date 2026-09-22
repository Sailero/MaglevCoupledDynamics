"""只画 DUE 张量：Y/U 的 (N,T) 矩阵和若干样本曲线。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from maglev_sim.due import load_due_dataset


def plot_due_dataset(
    npz_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """根据已生成的 DUE npz 写出两张图：矩阵热图、样本曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path = Path(npz_path)
    out_dir = Path(output_dir) if output_dir is not None else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_due_dataset(npz_path)
    Y = np.asarray(data["Y"], dtype=float)
    U = np.asarray(data["U"], dtype=float)
    t = np.asarray(data["t"], dtype=float)
    y_names = list(data["Y_names"])
    u_names = list(data["U_names"])
    n_samples, n_times, n_y = Y.shape

    written: list[Path] = []

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True)
    y_scale = 1e3
    for k, ax in enumerate(axes[0]):
        im = ax.imshow(y_scale * Y[:, :, k], aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(f"Y[{k}] {y_names[k]} [mm]\nshape ({n_samples}, {n_times})")
        ax.set_xlabel("time index")
        ax.set_ylabel("sample index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    u_scales = (1e3, 1.0)
    u_units = ("mm", "N")
    for k in range(2):
        ax = axes[1, k]
        im = ax.imshow(
            u_scales[k] * U[:, :, k],
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
        )
        ax.set_title(f"U[{k}] {u_names[k]} [{u_units[k]}]\nshape ({n_samples}, {n_times})")
        ax.set_xlabel("time index")
        ax.set_ylabel("sample index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.6,
        "DUE tensors\n"
        f"Y: {tuple(Y.shape)}  predict\n"
        f"U: {tuple(U.shape)}  known input\n"
        "Do not feed future Y as input.",
        transform=axes[1, 2].transAxes,
        va="center",
        fontsize=11,
        family="monospace",
    )
    fig.suptitle(f"DUE matrices  N={n_samples}  T={n_times}  row=sample  column=time")
    matrices_path = out_dir / f"{npz_path.stem}_matrices.png"
    fig.savefig(matrices_path, dpi=140)
    plt.close(fig)
    written.append(matrices_path)

    fig, axes = plt.subplots(n_y, 1, figsize=(10.5, 7.5), sharex=True, constrained_layout=True)
    pick = np.unique(np.linspace(0, n_samples - 1, num=min(6, n_samples), dtype=int))
    for k, ax in enumerate(axes):
        for i in pick:
            ax.plot(t, y_scale * Y[i, :, k], lw=1.1, label=f"sample {i}")
        ax.set_ylabel(f"{y_names[k]} [mm]")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, ncol=3, loc="upper right")
            ax.set_title("Y time series (subset of samples)")
    axes[-1].set_xlabel("t [s]")
    samples_path = out_dir / f"{npz_path.stem}_samples.png"
    fig.savefig(samples_path, dpi=140)
    plt.close(fig)
    written.append(samples_path)
    return written
