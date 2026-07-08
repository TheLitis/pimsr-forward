"""v3 TE+TM dataset: channel presence, mode independence, merge compat.

The 2-section builds run the SimPEG forward twice per section, so this
module is slow-ish (~1 min) but stays well inside CI budgets.
"""

import warnings

import h5py
import numpy as np
import pytest

pytest.importorskip("simpeg")

from pimsr_forward.dataset2d import build_dataset_2d, merge_shards  # noqa: E402

warnings.filterwarnings("ignore")

_TM_KEYS = (
    "obs_mt_log10_rho_tm", "obs_mt_phase_tm",
    "clean_mt_log10_rho_tm", "clean_mt_phase_tm",
)


@pytest.fixture(scope="module")
def small_ds(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds2d") / "ds.h5"
    build_dataset_2d(str(out), 2, seed=11, start_index=0)
    return str(out)


def test_tm_channels_present(small_ds):
    with h5py.File(small_ds) as f:
        for k in _TM_KEYS:
            assert k in f, k
            assert f[k].shape == f["obs_mt_log10_rho"].shape


def test_modes_differ_but_correlate(small_ds):
    # TE and TM must not be identical (2D structure) yet describe the same
    # section, so the clean log-rho fields should be strongly correlated.
    with h5py.File(small_ds) as f:
        te = f["clean_mt_log10_rho"][:]
        tm = f["clean_mt_log10_rho_tm"][:]
    assert not np.allclose(te, tm)
    r = np.corrcoef(te.ravel(), tm.ravel())[0, 1]
    # both modes see the same section, so they must correlate — but far
    # from perfectly (TM's galvanic sensitivity is the added information;
    # measured r ~ 0.7 on strongly 2D sections)
    assert r > 0.5


def test_tm_phase_in_quadrant(small_ds):
    with h5py.File(small_ds) as f:
        ph = f["clean_mt_phase_tm"][:]
    assert ph.min() > 0.0
    assert ph.max() < 90.0


def test_noise_independent_between_modes(small_ds):
    # static shifts are drawn separately per mode: the noisy/clean residual
    # patterns must differ between TE and TM.
    with h5py.File(small_ds) as f:
        res_te = f["obs_mt_log10_rho"][:] - f["clean_mt_log10_rho"][:]
        res_tm = f["obs_mt_log10_rho_tm"][:] - f["clean_mt_log10_rho_tm"][:]
    assert not np.allclose(res_te, res_tm)


def test_merge_v3_shards(small_ds, tmp_path):
    s2 = tmp_path / "s2.h5"
    build_dataset_2d(str(s2), 2, seed=11, start_index=2)
    merged = tmp_path / "merged.h5"
    n = merge_shards([small_ds, str(s2)], str(merged))
    assert n == 4
    with h5py.File(merged) as f:
        for k in _TM_KEYS:
            assert f[k].shape[0] == 4
