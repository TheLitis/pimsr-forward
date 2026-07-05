"""Gravity forward modeling: vertical attraction of right rectangular prisms.

Closed-form solution after Nagy (1966); the same expression used by
Harmonica and SimPEG. Evaluated for a profile of surface stations above a
layered/prism density model.

Output is in mGal (1 mGal = 1e-5 m/s^2). Densities are *contrasts* in kg/m^3
relative to a background, which is what a relative gravity survey senses.
"""

from __future__ import annotations

import numpy as np

G = 6.67430e-11  # m^3 kg^-1 s^-2
SI2MGAL = 1.0e5


def _nagy_kernel(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Nagy (1966) antiderivative x*log(y+r) + y*log(x+r) - z*arctan(xy/(zr))."""
    r = np.sqrt(x * x + y * y + z * z)
    out = np.zeros_like(r)
    # Guarded logs/arctans: terms vanish in the degenerate limits.
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = x * np.log(y + r)
        t2 = y * np.log(x + r)
        t3 = z * np.arctan2(x * y, z * r)
    for t in (t1, t2, t3):
        t[~np.isfinite(t)] = 0.0
    out = t1 + t2 - t3
    return out


def prism_gz(
    stations: np.ndarray,
    prism_bounds: np.ndarray,
    densities: np.ndarray,
) -> np.ndarray:
    """Vertical gravity (g_z, mGal) at stations from a set of prisms.

    Parameters
    ----------
    stations : (n_sta, 3) observation points (x, y, z), z positive *up*.
    prism_bounds : (n_prisms, 6) as (x1, x2, y1, y2, z1, z2), z positive up,
        z1 < z2 (z2 is the shallower face).
    densities : (n_prisms,) density contrast in kg/m^3.

    Returns
    -------
    gz : (n_sta,) in mGal, positive downward attraction (geophysical sign).
    """
    stations = np.atleast_2d(np.asarray(stations, dtype=np.float64))
    prism_bounds = np.atleast_2d(np.asarray(prism_bounds, dtype=np.float64))
    densities = np.asarray(densities, dtype=np.float64)

    if prism_bounds.shape[1] != 6:
        raise ValueError("prism_bounds must be (n_prisms, 6)")
    if densities.shape[0] != prism_bounds.shape[0]:
        raise ValueError("densities must match number of prisms")

    gz = np.zeros(stations.shape[0], dtype=np.float64)
    for p in range(prism_bounds.shape[0]):
        x1, x2, y1, y2, z1, z2 = prism_bounds[p]
        rho = densities[p]
        if rho == 0.0:
            continue
        # Shifted corner coordinates relative to each station.
        dx = np.stack([x1 - stations[:, 0], x2 - stations[:, 0]])  # (2, n)
        dy = np.stack([y1 - stations[:, 1], y2 - stations[:, 1]])
        dz = np.stack([z1 - stations[:, 2], z2 - stations[:, 2]])
        total = np.zeros(stations.shape[0])
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sign = (-1.0) ** (i + j + k)
                    total += sign * _nagy_kernel(dx[i], dy[j], dz[k])
        gz += G * rho * total
    return gz * SI2MGAL  # attraction from buried excess mass is positive


def layered_prism_bounds(
    thicknesses: np.ndarray,
    half_width: float = 5.0e4,
) -> np.ndarray:
    """Bounds for a stack of laterally wide prisms representing 1D layers.

    Layers start at z=0 (surface) going down; z axis positive up so layer
    tops/bottoms are negative. The terminating half-space is truncated at
    depth 2x the total stack for a finite prism approximation.
    """
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    tops = np.concatenate([[0.0], -np.cumsum(thicknesses)])
    bottom = tops[-1] - max(2.0 * np.sum(thicknesses), 1.0e4)
    z_edges = np.concatenate([tops, [bottom]])
    n_layers = len(z_edges) - 1
    bounds = np.zeros((n_layers, 6))
    for i in range(n_layers):
        bounds[i] = [-half_width, half_width, -half_width, half_width,
                     z_edges[i + 1], z_edges[i]]
    return bounds


def bouguer_slab_gz(thickness: float, density: float) -> float:
    """Infinite Bouguer slab: g = 2 pi G rho t, in mGal. Used as a test limit."""
    return 2.0 * np.pi * G * density * thickness * SI2MGAL
