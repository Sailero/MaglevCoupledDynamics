import numpy as np

from maglev_sim.plant.config import demo_config
from maglev_sim.plant.model import equilibrium, equilibrium_residuals
from maglev_sim.plant.state import layout_from_config


def test_i0_inside_current_limits():
    cfg = demo_config()
    assert cfg.magnet.I_min <= cfg.i0 <= cfg.magnet.I_max


def test_equilibrium_component_residuals():
    cfg = demo_config()
    layout = layout_from_config(cfg)
    resid = equilibrium_residuals(cfg)
    names = layout.names
    tol = {
        "z_c": 1e-10,
        "v_c": 1e-10,
        "z_m": 1e-10,
        "v_m": 1e-10,
        "i": 1e-8,
        "eta": 1e-10,
        "chi": 1e-10,
    }
    for name, value in zip(names, resid, strict=True):
        if name.startswith("q_") or name.startswith("p_"):
            limit = 1e-10
        else:
            limit = tol[name]
        assert abs(value) <= limit, f"{name} residual {value} > {limit}"


def test_equilibrium_uses_beam_static_deflection():
    cfg = demo_config()
    y = equilibrium(cfg)
    layout = layout_from_config(cfg)
    u = layout.unpack(y)
    assert u["z_c"] == u["z_m"]
    assert u["z_m"] > 0.0
    assert abs(u["i"] - cfg.i0) < 1e-12
