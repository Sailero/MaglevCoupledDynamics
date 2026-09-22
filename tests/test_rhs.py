import numpy as np

from maglev_sim.plant.config import demo_config
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.model import equilibrium, rhs
from maglev_sim.plant.state import layout_from_config


def test_force_signs_opposite_on_magnet_and_beam():
    cfg = demo_config()
    layout = layout_from_config(cfg)
    y = equilibrium(cfg)
    y_hi = y.copy()
    y_hi[layout.i] += 0.5
    exc = Excitation.from_config(cfg)
    f0 = rhs(0.0, y, cfg, exc)
    f1 = rhs(0.0, y_hi, cfg, exc)
    a_m = f1[layout.v_m]
    p1_dot = f1[layout.p_index(0)]
    assert a_m < f0[layout.v_m]
    assert a_m < 0.0
    assert p1_dot > f0[layout.p_index(0)]
    assert p1_dot > 0.0


def test_rhs_does_not_mutate_state():
    cfg = demo_config()
    y = equilibrium(cfg)
    snapshot = y.copy()
    exc = Excitation.from_config(cfg)
    rhs(0.0, y, cfg, exc)
    np.testing.assert_array_equal(y, snapshot)


def test_rhs_is_deterministic():
    cfg = demo_config()
    y = equilibrium(cfg)
    exc = Excitation.from_config(cfg)
    a = rhs(0.1, y, cfg, exc)
    b = rhs(0.1, y, cfg, exc)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)
