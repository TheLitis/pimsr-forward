import h5py
import numpy as np

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
    shape = (1, 2, 3, 4)
    response = MT3DResponse(np.ones(shape), np.full(shape, 45.0), ("xy", "yx"), 0.1)
    path = write_sample3d(tmp_path / "sample.h5", volume, response, [0.1], {"commit": "abc"})
    assert path.exists() and not (tmp_path / "sample.h5.part").exists()
    validate_sample3d(path)
    with h5py.File(path) as f:
        assert f["target/log10_resistivity"].shape == (6, 3, 4)
        assert f.attrs["schema_version"] == 1


def test_model_mapping_is_finite():
    # Exercise mapping without paying for a linear solve.
    from pimsr_forward.mt3d import MT3DForward

    solver = MT3DForward(frequencies=[0.1], station_x=[0], station_y=[0], cell_size=3000)
    model = solver.model_from_volume(_small_volume())
    assert model.shape == (solver.active.sum(),)
    assert np.isfinite(model).all()


def test_response_dataclass_contract():
    response = MT3DResponse(np.ones((1, 2, 1, 1)), np.ones((1, 2, 1, 1)), ("xy", "yx"), 1.0)
    assert response.modes == ("xy", "yx")
