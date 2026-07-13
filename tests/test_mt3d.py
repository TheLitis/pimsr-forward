import h5py
import numpy as np
import pytest

from pimsr_forward.dataset3d import validate_sample3d, write_sample3d
from pimsr_forward.mt3d import MT3DResponse
from pimsr_geogen.volume3d import VolumeGenerator


def _small_volume():
    return VolumeGenerator(
        seed=2,
        x_grid=np.linspace(-3000, 3000, 4),
        y_grid=np.linspace(-3000, 3000, 3),
        depth_grid=np.geomspace(10, 10000, 6),
    ).sample(1)


def test_atomic_hdf5_contract(tmp_path):
    volume = _small_volume()
    # Station grid deliberately differs from the geology grid: the survey
    # layout must never be forced to match model resolution.
    station_x, station_y = np.array([-1000.0, 1000.0]), np.array([0.0])
    shape = (1, 2, len(station_y), len(station_x))
    response = MT3DResponse(np.ones(shape), np.full(shape, 45.0), ("xy", "yx"), 0.1)
    path = write_sample3d(
        tmp_path / "sample.h5", volume, response, [0.1], station_x, station_y, {"commit": "abc"}
    )
    assert path.exists() and not (tmp_path / "sample.h5.part").exists()
    validate_sample3d(path)
    with h5py.File(path) as f:
        assert f["target/log10_resistivity"].shape == (6, 3, 4)
        assert f["observations/apparent_resistivity"].shape == shape
        assert f.attrs["schema_version"] == 2


def test_station_grid_mismatch_is_rejected(tmp_path):
    volume = _small_volume()
    station_x, station_y = np.array([-1000.0, 1000.0]), np.array([0.0])
    bad_shape = (1, 2, 5, 5)  # inconsistent with the declared station grid
    response = MT3DResponse(np.ones(bad_shape), np.full(bad_shape, 45.0), ("xy", "yx"), 0.1)
    with pytest.raises(ValueError, match="dimensions"):
        write_sample3d(tmp_path / "bad.h5", volume, response, [0.1], station_x, station_y)


def test_model_mapping_is_finite():
    # Exercise mapping without paying for a linear solve. The solver stack is
    # an optional dependency, matching the existing mt2d package contract.
    pytest.importorskip("discretize")
    pytest.importorskip("simpeg")
    from pimsr_forward.mt3d import MT3DForward

    solver = MT3DForward(frequencies=[0.1], station_x=[0], station_y=[0], cell_size=3000)
    model = solver.model_from_volume(_small_volume())
    assert model.shape == (solver.active.sum(),)
    assert np.isfinite(model).all()


def test_cli3d_resume_skips_valid_samples(tmp_path):
    from types import SimpleNamespace

    from pimsr_forward.cli3d import build

    volume = _small_volume()
    station = np.array([0.0])
    shape = (1, 2, 1, 1)
    response = MT3DResponse(np.ones(shape), np.full(shape, 45.0), ("xy", "yx"), 0.1)
    write_sample3d(tmp_path / "sample_0000000.h5", volume, response, [0.1], station, station)
    args = SimpleNamespace(
        out=str(tmp_path), start=0, count=1, seed=0, workers=1,
        nx=4, ny=3, nz=6, extent=6000.0, depth=10000.0, cell_size=3000.0,
        nstations=1, nfreq=1, fmin=0.1, fmax=0.1,
    )
    summary = build(args)
    assert summary == {"completed": 0, "skipped": 1, "output": str(tmp_path.resolve())}


def test_response_dataclass_contract():
    response = MT3DResponse(np.ones((1, 2, 1, 1)), np.ones((1, 2, 1, 1)), ("xy", "yx"), 1.0)
    assert response.modes == ("xy", "yx")
