from pathlib import Path

from maglev_sim.plant.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_demo_yaml_loads():
    cfg = load_config(ROOT / "configs" / "demo.yaml")
    cfg.validate()
    assert cfg.i0 > 0.0
    assert cfg.beam.n_modes >= 1
    assert cfg.meta.parameter_source == "demo_not_hs_maglev"
