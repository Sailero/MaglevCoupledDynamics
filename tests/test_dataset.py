import numpy as np

from maglev_sim.due import (
    NPZ_KEYS,
    U_NAMES,
    Y_NAMES,
    estimate_due_dataset,
    generate_due_dataset,
    load_due_dataset,
)


def test_due_tensors_only(tmp_path):
    out = tmp_path / "due.npz"
    generate_due_dataset(
        4, 80, out, dt=0.002, seed=3, z_m_std=1e-4, r_std=1e-4, verbose=False
    )
    raw = np.load(out, allow_pickle=True)
    assert set(raw.files) == set(NPZ_KEYS)
    data = load_due_dataset(out)
    assert data["Y"].shape == (4, 80, 3)
    assert data["U"].shape == (4, 80, 2)
    assert data["t"].shape == (80,)
    assert data["Y_names"] == Y_NAMES
    assert data["U_names"] == U_NAMES
    np.testing.assert_allclose(np.diff(data["t"]), 0.002)
    assert set(data["split"]) <= {"train", "val", "test"}
    assert np.std(data["U"][:, 0, 0]) > 0.0
    np.testing.assert_allclose(data["U"][:, :, 1], 0.0)
    np.testing.assert_allclose(data["U"][:, :1, 0], data["U"][:, -1:, 0])


def test_plot_due_dataset(tmp_path):
    from maglev_sim.plot import plot_due_dataset

    out = tmp_path / "due.npz"
    generate_due_dataset(3, 40, out, dt=0.002, seed=1, verbose=False)
    figs = plot_due_dataset(out, tmp_path)
    assert len(figs) == 2
    assert all(p.is_file() and p.stat().st_size > 0 for p in figs)


def test_estimate_shapes():
    info = estimate_due_dataset(100, 1000, dt=0.001, dtype="float32")
    assert info["Y_shape"] == (100, 1000, 3)
    assert info["U_shape"] == (100, 1000, 2)
    assert info["bytes_total"] == 100 * 1000 * 5 * 4
