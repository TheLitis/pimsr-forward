"""2D forward validation against the analytic 1D response."""

import numpy as np
import pytest

pytest.importorskip("simpeg")

from pimsr_forward.mt1d import mt1d_response  # noqa: E402
from pimsr_forward.mt2d import MT2DForward  # noqa: E402


@pytest.fixture(scope="module")
def fwd():
    return MT2DForward()


def _layered_section(fwd):
    """Laterally uniform 3-layer section on the standard grids."""
    from pimsr_geogen.model import DEFAULT_DEPTH_GRID
    from pimsr_geogen.section2d import DEFAULT_X_GRID, GeoSection2D

    z = DEFAULT_DEPTH_GRID
    log_res = np.empty((len(z), len(DEFAULT_X_GRID)))
    col = np.where(z <= 2000, np.log10(300.0), np.where(z <= 6000, 1.0, 3.0))
    log_res[:] = col[:, None]
    dens = np.full_like(log_res, 2500.0)
    return GeoSection2D(log10_res=log_res, density=dens)


def test_matches_1d_for_layered_earth(fwd):
    sec = _layered_section(fwd)
    rho_2d, ph_2d = fwd.response(sec)
    rho_1d, ph_1d = mt1d_response(
        np.array([300.0, 10.0, 1000.0]), np.array([2000.0, 4000.0]), fwd.periods
    )
    center = len(fwd.station_x) // 2
    rel = np.abs(rho_2d[:, center] - rho_1d) / rho_1d
    # ~5 % is grid-rasterisation error of the interfaces on the log-spaced
    # depth grid (13 % node spacing at 2 km), not solver error: the solver
    # matches the analytic response to 0.6 % when the mesh honours the
    # interfaces exactly (see validation in mt2d.py docstring).
    assert np.median(rel) < 0.08
    assert rel.max() < 0.15
    assert np.abs(ph_2d[:, center] - ph_1d).max() < 10.0


def test_lateral_contrast_is_visible(fwd):
    """A conductive body under one flank must split the station responses."""
    from pimsr_geogen.section2d import SectionGenerator

    gen = SectionGenerator(seed=0)
    sec = gen.sample(4, scenario="geothermal")
    rho_a, _ = fwd.response(sec)
    spread = np.log10(rho_a).std(axis=1).max()
    assert spread > 0.02


def test_output_shapes(fwd):
    sec = _layered_section(fwd)
    rho_a, phase = fwd.response(sec)
    assert rho_a.shape == (8, 12)
    assert phase.shape == (8, 12)
    assert np.all(rho_a > 0)
    assert np.all((phase > 0) & (phase < 90))
