"""Physics validation of the 1D MT kernel against analytic limits."""

import numpy as np
import pytest

from pimsr_forward.mt1d import (
    MU0,
    default_period_band,
    mt1d_impedance,
    mt1d_response,
)

PERIODS = default_period_band(32)


class TestHalfSpace:
    def test_apparent_resistivity_recovers_half_space(self):
        for rho in (1.0, 100.0, 5000.0):
            rho_a, phase = mt1d_response(
                np.array([rho]), np.array([]), PERIODS
            )
            np.testing.assert_allclose(rho_a, rho, rtol=1e-10)

    def test_half_space_phase_is_45_degrees(self):
        rho_a, phase = mt1d_response(np.array([100.0]), np.array([]), PERIODS)
        np.testing.assert_allclose(phase, 45.0, atol=1e-8)

    def test_impedance_magnitude_matches_definition(self):
        rho = 250.0
        Z = mt1d_impedance(np.array([rho]), np.array([]), PERIODS)
        omega = 2 * np.pi / PERIODS
        np.testing.assert_allclose(np.abs(Z), np.sqrt(omega * MU0 * rho), rtol=1e-10)


class TestTwoLayer:
    def test_short_periods_see_top_layer(self):
        # 10 Ohm*m over 1000 Ohm*m: highest frequency skin depth << thickness.
        rho_a, _ = mt1d_response(
            np.array([10.0, 1000.0]), np.array([5000.0]), np.array([1e-4])
        )
        np.testing.assert_allclose(rho_a, 10.0, rtol=1e-2)

    def test_long_periods_see_basement(self):
        rho_a, _ = mt1d_response(
            np.array([10.0, 1000.0]), np.array([100.0]), np.array([1e5])
        )
        np.testing.assert_allclose(rho_a, 1000.0, rtol=5e-2)

    def test_conductor_pulls_phase_above_45(self):
        # Resistor over conductor -> phase > 45 deg at intermediate periods.
        _, phase = mt1d_response(
            np.array([1000.0, 1.0]), np.array([2000.0]), np.array([1.0])
        )
        assert phase[0] > 50.0

    def test_equivalent_stack_matches_merged_layer(self):
        # Two identical adjacent layers == one merged layer.
        rho_a1, ph1 = mt1d_response(
            np.array([50.0, 50.0, 500.0]), np.array([1000.0, 2000.0]), PERIODS
        )
        rho_a2, ph2 = mt1d_response(
            np.array([50.0, 500.0]), np.array([3000.0]), PERIODS
        )
        np.testing.assert_allclose(rho_a1, rho_a2, rtol=1e-9)
        np.testing.assert_allclose(ph1, ph2, atol=1e-7)


class TestValidation:
    def test_rejects_negative_resistivity(self):
        with pytest.raises(ValueError):
            mt1d_impedance(np.array([-1.0]), np.array([]), PERIODS)

    def test_rejects_thickness_mismatch(self):
        with pytest.raises(ValueError):
            mt1d_impedance(np.array([10.0, 10.0]), np.array([]), PERIODS)
