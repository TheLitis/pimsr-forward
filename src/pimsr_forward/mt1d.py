"""Analytic 1D magnetotelluric forward modeling.

Implements the classic Wait (1954) impedance recursion for a stack of
horizontal, isotropic, homogeneous layers over a terminating half-space.
This is *exact* for the 1D case -- no discretization error -- which makes it
both a fast dataset generator and a differentiable-friendly reference
(the same recursion is re-implemented in torch inside ``pimsr-inversion``).

Conventions
-----------
- Time dependence ``exp(+i omega t)`` (engineering convention, matches SimPEG).
- Impedance ``Z = E_x / H_y`` in Ohm (SI).
- Apparent resistivity ``rho_a = |Z|^2 / (omega * mu0)``.
- Phase in degrees, ~45 deg over a half-space.
"""

from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi


def mt1d_impedance(
    resistivities: np.ndarray,
    thicknesses: np.ndarray,
    periods: np.ndarray,
) -> np.ndarray:
    """Surface impedance Z(T) for a layered half-space.

    Parameters
    ----------
    resistivities : (n_layers,) Ohm*m, last layer is the terminating half-space.
    thicknesses : (n_layers - 1,) m, thicknesses of the finite layers.
    periods : (n_periods,) s.

    Returns
    -------
    Z : (n_periods,) complex surface impedance in Ohm.
    """
    resistivities = np.asarray(resistivities, dtype=np.float64)
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)

    if resistivities.ndim != 1 or resistivities.size < 1:
        raise ValueError("resistivities must be a 1D array with >= 1 entries")
    if thicknesses.size != resistivities.size - 1:
        raise ValueError("need exactly n_layers - 1 thicknesses")
    if np.any(resistivities <= 0):
        raise ValueError("resistivities must be positive")
    if np.any(thicknesses <= 0):
        raise ValueError("thicknesses must be positive")

    omega = 2.0 * np.pi / periods  # (n_periods,)
    sigma = 1.0 / resistivities  # (n_layers,)

    # Propagation constant per layer/frequency: k = sqrt(i omega mu0 sigma)
    # with exp(+iwt) convention -> Re(k) > 0, decaying downward.
    # (n_periods, n_layers)
    k = np.sqrt(1j * omega[:, None] * MU0 * sigma[None, :])

    # Intrinsic impedance of each layer: Z_intr = i omega mu0 / k,
    # giving the standard +45 deg half-space phase.
    z_intr = 1j * omega[:, None] * MU0 / k

    # Recursion from the terminating half-space upward.
    z = z_intr[:, -1].copy()
    for j in range(resistivities.size - 2, -1, -1):
        zj = z_intr[:, j]
        kj = k[:, j]
        t = np.tanh(kj * thicknesses[j])
        z = zj * (z + zj * t) / (zj + z * t)
    return z


def impedance_to_apparent(
    Z: np.ndarray, periods: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert surface impedance to (apparent resistivity, phase in degrees)."""
    periods = np.asarray(periods, dtype=np.float64)
    omega = 2.0 * np.pi / periods
    rho_app = np.abs(Z) ** 2 / (omega * MU0)
    phase = np.degrees(np.arctan2(Z.imag, Z.real))
    return rho_app, phase


def mt1d_response(
    resistivities: np.ndarray,
    thicknesses: np.ndarray,
    periods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apparent resistivity and phase for a layered model. Convenience wrapper."""
    Z = mt1d_impedance(resistivities, thicknesses, periods)
    return impedance_to_apparent(Z, periods)


def default_period_band(n_periods: int = 24) -> np.ndarray:
    """Log-spaced period band 1e-3 .. 1e3 s (typical broadband MT survey)."""
    return np.logspace(-3.0, 3.0, n_periods)
