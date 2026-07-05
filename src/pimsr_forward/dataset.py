"""Dataset builder: geology models -> (observables, targets) HDF5.

Consumes a geology batch produced by ``pimsr-geogen`` and produces the
training file for ``pimsr-inversion``:

/obs_mt_log10_rho   (n, n_periods)  noisy log10 apparent resistivity
/obs_mt_phase       (n, n_periods)  noisy phase, degrees
/obs_gravity        (n, n_grav)     noisy relative gravity profile, mGal
/clean_mt_log10_rho (n, n_periods)  noise-free (for physics-loss debugging)
/clean_mt_phase     (n, n_periods)
/clean_gravity      (n, n_grav)
/target_log10_res   (n, n_grid)     fixed-depth-grid targets
/target_density     (n, n_grid)
/scenario           (n,) int8
/periods            (n_periods,)
/grav_offsets       (n_grav,)       station x-offsets, m
/depth_grid         (n_grid,)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pimsr_geogen.io import load_models

from .gravity import layered_prism_bounds, prism_gz
from .mt1d import default_period_band, mt1d_response
from .sensors import SensorModel

__all__ = ["build_dataset", "gravity_stations", "DEFAULT_GRAV_OFFSETS"]

#: Relative gravity profile: 16 stations along a 20 km line at the surface.
DEFAULT_GRAV_OFFSETS: np.ndarray = np.linspace(-1.0e4, 1.0e4, 16)
BACKGROUND_DENSITY = 2670.0  # kg/m^3, standard Bouguer reduction density


def gravity_stations(offsets: np.ndarray | None = None) -> np.ndarray:
    off = DEFAULT_GRAV_OFFSETS if offsets is None else np.asarray(offsets)
    stations = np.zeros((off.size, 3))
    stations[:, 0] = off
    stations[:, 2] = 1.0  # 1 m above surface avoids the singular corner
    return stations


def _model_gravity(model, stations: np.ndarray) -> np.ndarray:
    bounds = layered_prism_bounds(model.thicknesses)
    contrasts = model.densities - BACKGROUND_DENSITY
    gz = prism_gz(stations, bounds, contrasts)
    return gz - gz.mean()  # relative survey: remove absolute level


def build_dataset(
    geology_path: str | Path,
    out_path: str | Path,
    seed: int = 0,
    n_periods: int = 24,
    sensor: SensorModel | None = None,
) -> int:
    """Forward-model every geology model and write the training HDF5."""
    sensor = sensor or SensorModel()
    rng = np.random.default_rng(seed)
    periods = default_period_band(n_periods)
    stations = gravity_stations()
    models = load_models(geology_path)
    n = len(models)

    n_grav = stations.shape[0]
    obs_rho = np.zeros((n, n_periods), dtype=np.float32)
    obs_phase = np.zeros((n, n_periods), dtype=np.float32)
    obs_grav = np.zeros((n, n_grav), dtype=np.float32)
    clean_rho = np.zeros_like(obs_rho)
    clean_phase = np.zeros_like(obs_phase)
    clean_grav = np.zeros_like(obs_grav)
    tgt_res = np.zeros((n, 64), dtype=np.float32)
    tgt_den = np.zeros((n, 64), dtype=np.float32)
    scen = np.zeros(n, dtype=np.int8)

    from pimsr_geogen.generator import SCENARIOS
    from pimsr_geogen.model import DEFAULT_DEPTH_GRID

    for i, m in enumerate(models):
        rho_a, phase = mt1d_response(m.resistivities, m.thicknesses, periods)
        gz = _model_gravity(m, stations)

        clean_rho[i] = np.log10(rho_a)
        clean_phase[i] = phase
        clean_grav[i] = gz

        rho_n, phase_n = sensor.apply_mt(rho_a, phase, periods, rng)
        gz_n = sensor.apply_gravity(gz, rng)
        obs_rho[i] = np.log10(rho_n)
        obs_phase[i] = phase_n
        obs_grav[i] = gz_n

        grids = m.profile_on_grid()
        tgt_res[i] = grids["log10_resistivity"]
        tgt_den[i] = grids["density"]
        scen[i] = SCENARIOS.index(m.scenario)

    with h5py.File(out_path, "w") as f:
        for name, arr in [
            ("obs_mt_log10_rho", obs_rho),
            ("obs_mt_phase", obs_phase),
            ("obs_gravity", obs_grav),
            ("clean_mt_log10_rho", clean_rho),
            ("clean_mt_phase", clean_phase),
            ("clean_gravity", clean_grav),
            ("target_log10_res", tgt_res),
            ("target_density", tgt_den),
        ]:
            f.create_dataset(name, data=arr, compression="gzip")
        f.create_dataset("scenario", data=scen)
        f.create_dataset("periods", data=periods)
        f.create_dataset("grav_offsets", data=DEFAULT_GRAV_OFFSETS)
        f.create_dataset("depth_grid", data=DEFAULT_DEPTH_GRID)
        f.attrs["n_models"] = n
        f.attrs["seed"] = seed
        f.attrs["scenarios"] = ",".join(SCENARIOS)
    return n
