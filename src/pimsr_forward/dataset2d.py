"""2D profile dataset builder.

HDF5 layout (one row = one 2D section = one 12-station profile):

/obs_mt_log10_rho   (n, n_freq, n_st)  noisy log10 apparent resistivity (TE)
/obs_mt_phase       (n, n_freq, n_st)  noisy phase, degrees (TE)
/clean_mt_log10_rho (n, n_freq, n_st)
/clean_mt_phase     (n, n_freq, n_st)
/obs_mt_log10_rho_tm   (n, n_freq, n_st)  TM mode (v3+)
/obs_mt_phase_tm       (n, n_freq, n_st)
/clean_mt_log10_rho_tm (n, n_freq, n_st)
/clean_mt_phase_tm     (n, n_freq, n_st)
/target_log10_res   (n, n_z, n_x)      section on the fixed grid
/scenario           (n,)               int label index into SCENARIOS
/has_fault          (n,)               uint8
/frequencies        (n_freq,)
/station_x          (n_st,)
/x_grid             (n_x,)
/depth_grid         (n_z,)

Noise reuses the 1D SensorModel per station column (white + static shift +
correlated distortion), applied independently per station, which yields
laterally incoherent static shifts — the realistic case.
"""

from __future__ import annotations

import numpy as np

from .sensors import SensorModel

__all__ = ["build_dataset_2d", "merge_shards"]


def build_dataset_2d(
    path: str,
    n: int,
    seed: int = 0,
    start_index: int = 0,
    n_workers: int = 1,
) -> None:
    """Build a 2D profile dataset at ``path`` (HDF5)."""
    import h5py
    from pimsr_geogen.generator import SCENARIOS
    from pimsr_geogen.section2d import SectionGenerator

    from .mt2d import MT2DForward

    gen = SectionGenerator(seed=seed)
    fwd = MT2DForward()
    sensor = SensorModel()

    n_f, n_st = len(fwd.frequencies), len(fwd.station_x)
    n_z, n_x = len(gen.z), len(gen.x)
    periods = fwd.periods

    with h5py.File(path, "w") as f:
        obs_r = f.create_dataset("obs_mt_log10_rho", (n, n_f, n_st), dtype="f4")
        obs_p = f.create_dataset("obs_mt_phase", (n, n_f, n_st), dtype="f4")
        cl_r = f.create_dataset("clean_mt_log10_rho", (n, n_f, n_st), dtype="f4")
        cl_p = f.create_dataset("clean_mt_phase", (n, n_f, n_st), dtype="f4")
        obs_r_tm = f.create_dataset("obs_mt_log10_rho_tm", (n, n_f, n_st), dtype="f4")
        obs_p_tm = f.create_dataset("obs_mt_phase_tm", (n, n_f, n_st), dtype="f4")
        cl_r_tm = f.create_dataset("clean_mt_log10_rho_tm", (n, n_f, n_st), dtype="f4")
        cl_p_tm = f.create_dataset("clean_mt_phase_tm", (n, n_f, n_st), dtype="f4")
        tgt = f.create_dataset("target_log10_res", (n, n_z, n_x), dtype="f4")
        scen = f.create_dataset("scenario", (n,), dtype="i4")
        fault = f.create_dataset("has_fault", (n,), dtype="u1")
        f.create_dataset("frequencies", data=fwd.frequencies)
        f.create_dataset("station_x", data=fwd.station_x)
        f.create_dataset("x_grid", data=gen.x)
        f.create_dataset("depth_grid", data=gen.z)

        for i in range(n):
            idx = start_index + i
            sec = gen.sample(idx)
            rho_a, phase = fwd.response(sec)
            rho_tm, phase_tm = fwd.response_tm(sec)

            rng = np.random.default_rng([seed, 3, idx])
            rho_n = np.empty_like(rho_a)
            ph_n = np.empty_like(phase)
            rho_n_tm = np.empty_like(rho_tm)
            ph_n_tm = np.empty_like(phase_tm)
            # Per-section TM galvanic severity: most sections mildly worse
            # than TE, a tail of strongly distorted ones (real rows I/K).
            tm_shift_sigma = float(rng.uniform(0.15, 0.40))
            tm_distort_hi = float(np.exp(rng.uniform(np.log(0.25), np.log(0.60))))
            for j in range(n_st):
                rho_n[:, j], ph_n[:, j] = sensor.apply_mt(
                    rho_a[:, j], phase[:, j], periods, rng
                )
                # independent noise draw per mode: galvanic static shifts
                # differ between xy and yx in real data, and yx is boosted
                rho_n_tm[:, j], ph_n_tm[:, j] = sensor.apply_mt(
                    rho_tm[:, j], phase_tm[:, j], periods, rng,
                    shift_sigma=tm_shift_sigma, distort_hi=tm_distort_hi,
                )

            obs_r[i] = np.log10(rho_n)
            obs_p[i] = ph_n
            cl_r[i] = np.log10(rho_a)
            cl_p[i] = phase
            obs_r_tm[i] = np.log10(rho_n_tm)
            obs_p_tm[i] = ph_n_tm
            cl_r_tm[i] = np.log10(rho_tm)
            cl_p_tm[i] = phase_tm
            tgt[i] = sec.log10_res
            scen[i] = SCENARIOS.index(sec.scenario)
            fault[i] = int(sec.has_fault)
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{n}] sections done", flush=True)


_ROW_KEYS = (
    "obs_mt_log10_rho", "obs_mt_phase", "clean_mt_log10_rho",
    "clean_mt_phase", "target_log10_res", "scenario", "has_fault",
    "obs_mt_log10_rho_tm", "obs_mt_phase_tm",
    "clean_mt_log10_rho_tm", "clean_mt_phase_tm",
)
_META_KEYS = ("frequencies", "station_x", "x_grid", "depth_grid")


def merge_shards(shard_paths: list[str], out_path: str) -> int:
    """Concatenate shard HDF5 files (in the given order) into one dataset.

    Row keys are taken from the intersection with the first shard, so the
    merger works for both legacy TE-only and v3 TE+TM shard layouts.
    """
    import h5py

    total = 0
    row_keys: tuple[str, ...] = _ROW_KEYS
    with h5py.File(out_path, "w") as out:
        for si, sp in enumerate(shard_paths):
            with h5py.File(sp, "r") as src:
                n = src["scenario"].shape[0]
                if si == 0:
                    row_keys = tuple(k for k in _ROW_KEYS if k in src)
                    for k in _META_KEYS:
                        out.create_dataset(k, data=src[k][:])
                    for k in row_keys:
                        shape = (0,) + src[k].shape[1:]
                        maxshape = (None,) + src[k].shape[1:]
                        out.create_dataset(k, shape, maxshape=maxshape, dtype=src[k].dtype)
                for k in row_keys:
                    ds = out[k]
                    ds.resize(total + n, axis=0)
                    ds[total : total + n] = src[k][:]
                total += n
    return total
