import numpy as np
from scipy.integrate import solve_ivp

from maglev_sim.plant.config import BeamConfig
from maglev_sim.plant.physics.beam import BeamModes, static_midspan_point_load


def test_single_mode_undamped_oscillator():
    beam = BeamConfig(n_modes=1, zeta=0.0)
    modes = BeamModes.from_config(beam)
    q0 = 1.0e-3
    omega = float(modes.omega[0])

    def fun(t, y):
        q, p = y
        qd, pd = modes.modal_rhs(
            np.array([q]), np.array([p]), force_at_magnet=0.0, xi=beam.L / 2, x_f=beam.L / 2, P_ext=0.0
        )
        return [qd[0], pd[0]]

    t = np.linspace(0.0, 0.4, 401)
    sol = solve_ivp(fun, (0.0, t[-1]), [q0, 0.0], t_eval=t, rtol=1e-8, atol=1e-12)
    q_exact = q0 * np.cos(omega * t)
    err = np.max(np.abs(sol.y[0] - q_exact))
    assert err < 2e-6


def test_static_midspan_converges_to_closed_form():
    beam = BeamConfig()
    P = 1.0e4
    exact = static_midspan_point_load(P, beam)
    x = 0.5 * beam.L
    errors = []
    for n in (1, 3, 7, 15, 31):
        cfg = BeamConfig(n_modes=n)
        modes = BeamModes.from_config(cfg)
        q = modes.static_modal_amplitudes(P, x)
        w = modes.displacement(x, q)
        errors.append(abs(w - exact) / abs(exact))
    assert errors[-1] < 1e-3
    assert errors[-1] < errors[0]
