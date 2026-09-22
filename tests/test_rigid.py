from maglev_sim.plant.config import demo_config
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.model import initial_state
from maglev_sim.plant.simulate import simulate
from maglev_sim.plant.state import layout_from_config
import numpy as np


def test_rigid_track_freezes_beam():
    cfg = demo_config()
    cfg.output.t_end = 0.4
    cfg.beam.enabled = False
    cfg.initial.delta = {"z_m": 2e-4}
    y0 = initial_state(cfg)
    exc = Excitation.from_config(cfg)
    traj = simulate(cfg, y0, exc, cfg.output_times())
    layout = layout_from_config(cfg)
    q0 = y0[layout.q_index(0)]
    np.testing.assert_allclose(traj.q[:, 0], q0, atol=1e-14)
    np.testing.assert_allclose(traj.beam_displacement_at_magnet, 0.0, atol=1e-14)
