"""Coarse SimPEG 3D MT feasibility forward model.

This is deliberately a smoke-scale primary/secondary solver, not the final
production mesh.  It emits both off-diagonal impedance modes in an HDF5-ready
shape and reports solve timing so cloud sizing is based on measurements.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["MT3DForward", "MT3DResponse"]

_AIR_SIGMA = 1e-8


@dataclass(frozen=True)
class MT3DResponse:
    apparent_resistivity: np.ndarray  # (frequency, mode, y_station, x_station)
    phase: np.ndarray
    modes: tuple[str, str]
    wall_time_s: float


class MT3DForward:
    """Reusable coarse 3D natural-source solver for :class:`GeoVolume3D`."""

    def __init__(self, frequencies=None, station_x=None, station_y=None, cell_size=2000.0):
        import discretize
        from simpeg import maps
        from simpeg.electromagnetics import natural_source as nsem

        self.frequencies = np.asarray([0.1] if frequencies is None else frequencies, float)
        self.station_x = np.asarray([-2000.0, 2000.0] if station_x is None else station_x, float)
        self.station_y = np.asarray([-2000.0, 2000.0] if station_y is None else station_y, float)
        # Symmetric lateral padding, six subsurface cells and two air cells.
        hxy = [(cell_size, 1, -1.5), (cell_size, 4), (cell_size, 1, 1.5)]
        hz = [(cell_size, 2, -1.5), (cell_size, 6), (cell_size, 2, 1.5)]
        temp = discretize.TensorMesh([hxy, hxy, hz])
        sub_height = float(np.sum(temp.h[2][:-2]))
        self.mesh = discretize.TensorMesh(
            [hxy, hxy, hz], x0=[-temp.h[0].sum() / 2, -temp.h[1].sum() / 2, -sub_height]
        )
        self.active = self.mesh.cell_centers[:, 2] < 0
        xx, yy = np.meshgrid(self.station_x, self.station_y)
        locations = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
        receivers = []
        for orientation in ("xy", "yx"):
            receivers += [
                nsem.receivers.Impedance(locations, orientation=orientation, component="apparent_resistivity"),
                nsem.receivers.Impedance(locations, orientation=orientation, component="phase"),
            ]
        sources = [nsem.sources.PlanewaveXYPrimary(receivers, f) for f in self.frequencies]
        survey = nsem.survey.Survey(sources)
        primary = np.where(self.active, 1e-2, _AIR_SIGMA)
        inject = maps.InjectActiveCells(self.mesh, self.active, np.log(_AIR_SIGMA))
        self.simulation = nsem.simulation.Simulation3DPrimarySecondary(
            self.mesh, survey=survey, sigmaPrimary=primary, sigmaMap=maps.ExpMap(self.mesh) * inject
        )

    def model_from_volume(self, volume) -> np.ndarray:
        cc = self.mesh.cell_centers[self.active]
        ix = np.clip(np.searchsorted(volume.x_grid, cc[:, 0]), 0, len(volume.x_grid) - 1)
        iy = np.clip(np.searchsorted(volume.y_grid, cc[:, 1]), 0, len(volume.y_grid) - 1)
        iz = np.clip(np.searchsorted(volume.depth_grid, -cc[:, 2]), 0, len(volume.depth_grid) - 1)
        return np.log(10.0 ** (-volume.log10_res[iz, iy, ix]))

    def response(self, volume) -> MT3DResponse:
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.simulation.dpred(self.model_from_volume(volume))
        nf, ny, nx = len(self.frequencies), len(self.station_y), len(self.station_x)
        shaped = data.reshape(nf, 2, 2, ny, nx)  # frequency, mode, component, y, x
        # Fold SimPEG's orientation-dependent phase conventions to the EMTF
        # 0..180 range. Layered-earth off-diagonal phases then lie in 0..90.
        phase = np.mod(shaped[:, :, 1], 180.0)
        return MT3DResponse(
            shaped[:, :, 0], phase, ("xy", "yx"), time.perf_counter() - started
        )
