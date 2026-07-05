"""The correlated distortion must match the measured real-data statistics."""

import numpy as np

from pimsr_forward.sensors import SensorModel, _ar1_curve


def test_ar1_statistics():
    rng = np.random.default_rng(0)
    x = _ar1_curve(20000, 0.46, rng)
    assert abs(x.std() - 1.0) < 0.05
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(lag1 - 0.46) < 0.05


def test_distortion_reaches_real_amplitudes():
    """Pooled residual std across many synthetic stations should land in the
    real-data range (pooled 0.085, per-station 0.02-0.26)."""
    sm = SensorModel()
    rng = np.random.default_rng(1)
    periods = np.logspace(-2, 3, 40)
    rho = np.full(40, 100.0)
    ph = np.full(40, 45.0)
    stds = []
    for _ in range(300):
        r, _p = sm.apply_mt(rho, ph, periods, rng, static_shift=False)
        d = np.log10(r / rho)
        stds.append((d - d.mean()).std())
    pooled = float(np.mean(stds))
    assert 0.04 < pooled < 0.15
    assert min(stds) > 0.01 and max(stds) < 0.4


def test_distortion_disabled():
    sm = SensorModel(distort_log10rho_hi=0.0)
    rng = np.random.default_rng(2)
    periods = np.logspace(-2, 3, 40)
    rho = np.full(40, 100.0)
    ph = np.full(40, 45.0)
    r, _ = sm.apply_mt(rho, ph, periods, rng, static_shift=False)
    d = np.log10(r / rho)
    assert (d - d.mean()).std() < 0.05  # only the 3% floor remains
