from pathlib import Path

import pytest

from maglev_sim.plant.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_config():
    return load_config(ROOT / "configs" / "demo.yaml")
