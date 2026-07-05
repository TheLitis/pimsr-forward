"""Physics validation of the prism gravity kernel."""

import numpy as np

from pimsr_forward.gravity import (
    bouguer_slab_gz,
    layered_prism_bounds,
    prism_gz,
)


class TestBouguerLimit:
    def test_wide_slab_approaches_bouguer(self):
        """A very wide, thin prism under the station ~ infinite slab."""
        thickness, rho = 100.0, 300.0
        station = np.array([[0.0, 0.0, 0.0]])
        expected = bouguer_slab_gz(thickness, rho)
        errs = []
        for hw in (5e5, 5e6):
            bounds = np.array([[-hw, hw, -hw, hw, -1000.0 - thickness, -1000.0]])
            gz = prism_gz(station, bounds, np.array([rho]))
            errs.append(abs(gz[0] - expected) / expected)
        # converges to the slab limit ~ 1/width
        assert errs[-1] < 3e-4
        assert errs[1] < errs[0] / 5.0

    def test_positive_contrast_gives_positive_anomaly(self):
        bounds = np.array([[-1e3, 1e3, -1e3, 1e3, -2000.0, -500.0]])
        station = np.array([[0.0, 0.0, 0.0]])
        gz = prism_gz(station, bounds, np.array([200.0]))
        assert gz[0] > 0


class TestSymmetry:
    def test_lateral_symmetry(self):
        bounds = np.array([[-1e3, 1e3, -1e3, 1e3, -3000.0, -1000.0]])
        stations = np.array([[-5e3, 0, 0], [5e3, 0, 0]], dtype=float)
        gz = prism_gz(stations, bounds, np.array([250.0]))
        np.testing.assert_allclose(gz[0], gz[1], rtol=1e-12)

    def test_superposition(self):
        b1 = np.array([[-1e3, 0, -1e3, 1e3, -2000.0, -500.0]])
        b2 = np.array([[0, 1e3, -1e3, 1e3, -2000.0, -500.0]])
        both = np.vstack([b1, b2])
        station = np.array([[100.0, 0.0, 0.0]])
        rho = np.array([300.0])
        gz_sum = prism_gz(station, b1, rho) + prism_gz(station, b2, rho)
        gz_both = prism_gz(station, both, np.array([300.0, 300.0]))
        np.testing.assert_allclose(gz_both, gz_sum, rtol=1e-12)

    def test_decay_with_distance(self):
        bounds = np.array([[-500, 500, -500, 500, -1500.0, -1000.0]])
        stations = np.array([[0, 0, 0], [2e4, 0, 0]], dtype=float)
        gz = prism_gz(stations, bounds, np.array([300.0]))
        assert abs(gz[0]) > 10 * abs(gz[1])


class TestLayeredBounds:
    def test_bounds_are_contiguous(self):
        th = np.array([200.0, 800.0, 3000.0])
        bounds = layered_prism_bounds(th)
        # top of first layer at 0
        assert bounds[0, 5] == 0.0
        for i in range(len(bounds) - 1):
            assert bounds[i, 4] == bounds[i + 1, 5]
