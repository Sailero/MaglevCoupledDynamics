from maglev_sim.plant.config import demo_config
from maglev_sim.plant.control.gap_pid import GapPIDController


def test_anti_windup_stops_integral_when_saturating():
    cfg = demo_config()
    ctrl = GapPIDController()
    error = 0.02
    out = ctrl.command_and_derivs(error, cfg.i0, eta=0.0, chi=0.0, config=cfg)
    assert out.saturated
    assert out.i_cmd == cfg.magnet.I_max
    assert out.eta_dot == 0.0


def test_integral_runs_when_unsaturated():
    cfg = demo_config()
    ctrl = GapPIDController()
    error = 1.0e-5
    out = ctrl.command_and_derivs(error, cfg.i0, eta=0.0, chi=0.0, config=cfg)
    assert not out.saturated
    assert out.eta_dot == error
