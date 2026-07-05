"""Sensor noise statistics and end-to-end dataset build."""

import h5py
import numpy as np

from pimsr_forward.dataset import build_dataset
from pimsr_forward.mt1d import default_period_band
from pimsr_forward.sensors import SensorModel


class TestSensorModel:
    def test_mt_noise_is_unbiased_in_log(self):
        sensor = SensorModel(static_shift_sigma=0.0)
        rng = np.random.default_rng(1)
        periods = default_period_band(24)
        rho = np.full(24, 100.0)
        phase = np.full(24, 45.0)
        logs = []
        for _ in range(2000):
            r, _ = sensor.apply_mt(rho, phase, periods, rng, static_shift=False)
            logs.append(np.log(r))
        mean_log = np.mean(logs)
        assert abs(mean_log - np.log(100.0)) < 0.005

    def test_dead_band_noisier(self):
        sensor = SensorModel()
        periods = default_period_band(24)
        sig = sensor.mt_sigma(np.full(24, 100.0), periods)
        dead = (periods >= 0.1) & (periods <= 10.0)
        assert sig[dead].min() > sig[~dead].max()

    def test_gravity_noise_amplitude(self):
        sensor = SensorModel()
        rng = np.random.default_rng(2)
        gz = np.zeros(16)
        noisy = np.stack([sensor.apply_gravity(gz, rng) for _ in range(500)])
        # Total std should be on the order of white + drift, well under 0.5 mGal
        assert 0.01 < noisy.std() < 0.3


class TestDatasetBuild:
    def test_end_to_end(self, tmp_path):
        from pimsr_geogen.generator import GeologyGenerator
        from pimsr_geogen.io import save_models

        geo = tmp_path / "geo.h5"
        out = tmp_path / "ds.h5"
        save_models(GeologyGenerator(seed=7).sample_batch(8), geo)
        n = build_dataset(geo, out, seed=1)
        assert n == 8
        with h5py.File(out) as f:
            assert f["obs_mt_log10_rho"].shape == (8, 24)
            assert f["obs_gravity"].shape == (8, 16)
            assert f["target_log10_res"].shape == (8, 64)
            # Noise-free channels differ from noisy ones
            assert not np.allclose(
                f["obs_mt_log10_rho"][:], f["clean_mt_log10_rho"][:]
            )
            # Gravity profiles are zero-mean (relative survey)
            np.testing.assert_allclose(
                f["clean_gravity"][:].mean(axis=1), 0.0, atol=1e-8
            )
