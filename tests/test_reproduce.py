import numpy as np

from maglev_sim.plant.config import demo_config
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.model import initial_state
from maglev_sim.plant.simulate import resume, simulate


def test_identical_runs_repeat():
    cfg = demo_config()
    cfg.output.t_end = 0.3
    cfg.initial.delta = {"z_m": 1e-4}
    y0 = initial_state(cfg)
    exc = Excitation.from_config(cfg)
    t = cfg.output_times()
    a = simulate(cfg, y0, exc, t)
    b = simulate(cfg, y0, exc, t)
    np.testing.assert_allclose(a.state, b.state, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.gap_raw, b.gap_raw, rtol=1e-12, atol=1e-12)


def test_resume_matches_uninterrupted():
    cfg = demo_config()
    cfg.output.t_end = 0.4
    cfg.initial.delta = {"z_m": 1e-4}
    y0 = initial_state(cfg)
    exc = Excitation.from_config(cfg)
    t = cfg.output_times()
    full = simulate(cfg, y0, exc, t)
    mid = 0.2
    idx = int(np.argmin(np.abs(full.t - mid)))
    first = simulate(cfg, y0, exc, t[t <= full.t[idx] + 1e-15])
    rest_times = t[t >= first.t[-1] - 1e-15]
    cont = resume(cfg, first.state[-1], exc, rest_times)
    joined_t = np.concatenate([first.t[:-1], cont.t])
    joined_y = np.vstack([first.state[:-1], cont.state])
    ref_idx = np.searchsorted(full.t, joined_t)
    np.testing.assert_allclose(joined_y, full.state[ref_idx], rtol=2e-7, atol=2e-9)
