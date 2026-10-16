import numpy as np
import pytest

from maglev_sim.plant.config import demo_config
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.model import initial_state, observe


def temporal_config():
    cfg = demo_config()
    cfg.output.t_start, cfg.output.t_end = 1., 2.
    cfg.irregularity.type = 'time_series'
    cfg.irregularity.t = [1., 1.5, 2.]
    cfg.irregularity.r = [0.0001, 0.0003, 0.0002]
    return cfg


def test_temporal_interpolation_initial_equilibrium_and_observation():
    cfg = temporal_config()
    exc = Excitation.from_config(cfg)
    assert exc.irregularity_value(9., 1.25) == pytest.approx(.0002)
    y = initial_state(cfg, excitation=exc)
    obs = observe(1., y, cfg, exc)
    assert obs.gap_raw == pytest.approx(cfg.magnet.delta_0)
    assert obs.irregularity == .0001
    with pytest.raises(ValueError, match='time'):
        exc.irregularity_value(9.)


@pytest.mark.parametrize('times,values', [([1, 1, 2], [0, 1, 2]), ([2, 1], [0, 1]),
    ([1], [0]), ([[1, 2]], [[0, 1]]), ([1, np.nan], [0, 1]),
    ([1, 2], [0, np.inf]), ([1, 2], [0]), ([1.1, 2], [0, 1]), ([1, 1.9], [0, 1])])
def test_bad_time_series_rejected(times, values):
    cfg = temporal_config()
    cfg.irregularity.t, cfg.irregularity.r = times, values
    with pytest.raises(ValueError):
        Excitation.from_config(cfg)


def test_spatial_modes_unchanged_and_equilibrium_does_not_build_random_force():
    cfg = demo_config()
    cfg.irregularity.type, cfg.irregularity.amplitude = 'spatial_sine', .0002
    cfg.force.type, cfg.force.amplitude = 'precomputed', 100.
    exc = Excitation.from_config(cfg, seed=10)
    assert exc.irregularity_value(2.) == pytest.approx(.0002)
    assert exc.irregularity_value(2., .3) == exc.irregularity_value(2.)
    assert np.isfinite(initial_state(cfg)).all()


def test_force_coverage_and_interval_extension():
    cfg = temporal_config()
    cfg.force.type, cfg.force.t, cfg.force.P = 'precomputed', [1., 1.9], [0., 1.]
    with pytest.raises(ValueError, match='cover'):
        Excitation.from_config(cfg)
    cfg.force.t = [1., 2.]
    exc = Excitation.from_config(cfg)
    with pytest.raises(ValueError, match='cover'):
        exc.validate_interval(1., 2.1)


def test_missing_force_values_rejected_when_times_given():
    cfg = demo_config()
    cfg.force.type, cfg.force.t, cfg.force.P, cfg.force.seed = 'precomputed', [0., 5.], None, 3
    with pytest.raises(ValueError, match='both'):
        Excitation.from_config(cfg)
