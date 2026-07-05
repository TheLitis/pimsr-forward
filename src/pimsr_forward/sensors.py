"""Sensor and noise models for synthetic observables.

MT noise model
--------------
Field MT impedance estimates carry errors that are usually quoted as a
fraction of |Z| (processing error floor, typically 2-5 %) plus period-dependent
scatter in the dead band (0.1-10 s). Static shift -- a frequency-independent
multiplicative offset of apparent resistivity caused by near-surface
galvanic distortion -- is applied per station as log-uniform.

Gravity noise model
-------------------
Modern relative gravimeters (e.g. CG-6) achieve ~5-10 uGal repeatability;
survey-grade residual errors after drift correction are ~0.02-0.05 mGal.
We model white noise plus a smooth residual drift along the profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorModel:
    """Noise configuration for one synthetic 'survey'."""

    mt_rel_floor: float = 0.03          # 3 % of |rho_a|
    mt_phase_floor_deg: float = 1.0     # absolute phase scatter
    mt_dead_band_extra: float = 0.02    # extra rel. noise inside dead band
    static_shift_sigma: float = 0.15    # sigma of log10 static shift
    grav_white_mgal: float = 0.03       # white noise per station
    grav_drift_mgal: float = 0.05       # amplitude of smooth residual drift

    def apply_mt(
        self,
        rho_app: np.ndarray,
        phase: np.ndarray,
        periods: np.ndarray,
        rng: np.random.Generator,
        static_shift: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return noisy (rho_app, phase). Shapes preserved."""
        periods = np.asarray(periods)
        rel = np.full_like(rho_app, self.mt_rel_floor)
        dead = (periods >= 0.1) & (periods <= 10.0)
        rel = rel + self.mt_dead_band_extra * dead

        rho_noisy = rho_app * np.exp(rng.normal(0.0, rel))
        if static_shift:
            shift = 10.0 ** rng.normal(0.0, self.static_shift_sigma)
            rho_noisy = rho_noisy * shift
        phase_noisy = phase + rng.normal(0.0, self.mt_phase_floor_deg, phase.shape)
        return rho_noisy, phase_noisy

    def apply_gravity(
        self, gz: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return noisy gravity profile (white + smooth drift)."""
        gz = np.asarray(gz, dtype=np.float64)
        n = gz.size
        white = rng.normal(0.0, self.grav_white_mgal, n)
        # Smooth drift: low-order random polynomial along the profile.
        x = np.linspace(-1.0, 1.0, n)
        coeffs = rng.normal(0.0, self.grav_drift_mgal, 3)
        drift = coeffs[0] * x + coeffs[1] * (x**2 - 1.0 / 3.0) + coeffs[2] * 0.5 * x**3
        return gz + white + drift

    def mt_sigma(self, rho_app: np.ndarray, periods: np.ndarray) -> np.ndarray:
        """Expected 1-sigma of log(rho_a) used for chi^2 normalization."""
        periods = np.asarray(periods)
        rel = np.full_like(rho_app, self.mt_rel_floor)
        rel = rel + self.mt_dead_band_extra * ((periods >= 0.1) & (periods <= 10.0))
        return rel
