from maglev_sim.plant.config import demo_config
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.integrate import GAP_MIN, LEFT_BEAM
from maglev_sim.plant.model import initial_state
from maglev_sim.plant.simulate import simulate


def test_gap_floor_does_not_rewrite_gap_raw():
    cfg = demo_config()
    cfg.output.t_end = 0.2
    cfg.controller.Kp = 0.0
    cfg.controller.Ki = 0.0
    cfg.controller.Kd = 0.0
    cfg.magnet.gap_min = 0.002
    cfg.initial.delta = {"z_m": -2.0e-4}
    y0 = initial_state(cfg)
    exc = Excitation.from_config(cfg)
    traj = simulate(cfg, y0, exc, cfg.output_times())
    assert traj.termination_reason in {GAP_MIN, "completed"}
    if traj.termination_reason == GAP_MIN:
        assert traj.gap_raw[-1] <= cfg.magnet.gap_min + 2e-4
        assert not (abs(traj.gap_raw[-1] - cfg.magnet.delta_floor) < 1e-12)


def test_moving_point_stops_at_beam_end():
    cfg = demo_config()
    cfg.motion.mode = "moving"
    cfg.motion.xi = 2.0
    cfg.motion.v = 5.0
    cfg.output.dt = 0.002
    cfg.output.t_end = 6.0
    y0 = initial_state(cfg)
    exc = Excitation.from_config(cfg)
    traj = simulate(cfg, y0, exc, cfg.output_times())
    assert traj.termination_reason == LEFT_BEAM
    assert not traj.completed
    assert traj.xi[-1] >= cfg.beam.L - 2e-3 or traj.xi[-1] <= 2e-3
