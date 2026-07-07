"""2D magnetotelluric forward modelling via SimPEG (TE + TM modes).

Validated against the analytic 1D Wait recursion for layered models:
median apparent-resistivity error 0.4 % (TE) / 0.13 % (TM) across
1e-2..1e2 s (max 7 % at the longest period, controlled by the depth of
the padding region). TM phase comes out of SimPEG's yx receiver already
in the first-quadrant convention (offset 0), unlike TE's xy (+180).

The mesh is built once and reused across models — only the conductivity
vector changes — which keeps the per-model cost at ~0.5 s for
8 frequencies x 12 stations on a laptop-class CPU.

SimPEG is an optional dependency: ``pip install pimsr-forward[mt2d]``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["MT2DForward", "DEFAULT_FREQUENCIES", "DEFAULT_STATION_X"]

#: 8 frequencies, 0.01..10 Hz — the band where 2D effects matter most and
#: the SimPEG mesh stays affordable.
DEFAULT_FREQUENCIES: np.ndarray = np.logspace(-2, 1, 8)

#: 12 stations spanning the central 16 km of the 24 km section.
DEFAULT_STATION_X: np.ndarray = np.linspace(-8000.0, 8000.0, 12)

_AIR_SIGMA = 1e-8


@dataclass(frozen=True)
class _Mesh2D:
    mesh: object
    active_cc: np.ndarray  # (n_active, 2) subsurface cell centers
    active_idx: np.ndarray  # indices of subsurface cells
    air_idx: np.ndarray


def _build_mesh() -> _Mesh2D:
    import discretize

    dx, dz = 500.0, 100.0
    hx = [(dx, 10, -1.5), (dx, 48), (dx, 10, 1.5)]
    hz_sub = [(dz, 16, -1.5), (dz, 70)]
    hz_air = [(dz, 12, 1.6)]
    hz = hz_sub + hz_air
    tmp = discretize.TensorMesh([hx, hz])
    sub_h = float(np.sum(tmp.h[1][: 16 + 70]))
    mesh = discretize.TensorMesh([hx, hz], x0=[-float(np.sum(tmp.h[0])) / 2, -sub_h])
    cc = mesh.cell_centers
    sub = cc[:, 1] < 0
    return _Mesh2D(
        mesh=mesh,
        active_cc=cc[sub],
        active_idx=np.flatnonzero(sub),
        air_idx=np.flatnonzero(~sub),
    )


class MT2DForward:
    """Reusable 2D TE-mode MT simulator for :class:`GeoSection2D` objects.

    Parameters
    ----------
    frequencies : Hz.
    station_x : station positions along the profile, m.
    """

    def __init__(
        self,
        frequencies: np.ndarray | None = None,
        station_x: np.ndarray | None = None,
    ) -> None:
        from simpeg import maps
        from simpeg.electromagnetics import natural_source as nsem

        self.frequencies = (
            DEFAULT_FREQUENCIES.copy() if frequencies is None else np.asarray(frequencies, float)
        )
        self.station_x = (
            DEFAULT_STATION_X.copy() if station_x is None else np.asarray(station_x, float)
        )
        self._m = _build_mesh()

        rx_locs = np.c_[self.station_x, np.zeros_like(self.station_x)]
        rx = [
            nsem.receivers.Impedance(rx_locs, orientation="xy", component="apparent_resistivity"),
            nsem.receivers.Impedance(rx_locs, orientation="xy", component="phase"),
        ]
        srcs = [nsem.sources.Planewave(rx, frequency=f) for f in self.frequencies]
        survey = nsem.survey.Survey(srcs)
        self._sim = nsem.simulation.Simulation2DElectricField(
            self._m.mesh, survey=survey, sigmaMap=maps.IdentityMap(self._m.mesh)
        )

        # TM mode: H-field formulation with yx impedance receivers.
        rx_tm = [
            nsem.receivers.Impedance(rx_locs, orientation="yx", component="apparent_resistivity"),
            nsem.receivers.Impedance(rx_locs, orientation="yx", component="phase"),
        ]
        srcs_tm = [nsem.sources.Planewave(rx_tm, frequency=f) for f in self.frequencies]
        survey_tm = nsem.survey.Survey(srcs_tm)
        self._sim_tm = nsem.simulation.Simulation2DMagneticField(
            self._m.mesh, survey=survey_tm, sigmaMap=maps.IdentityMap(self._m.mesh)
        )

    # ------------------------------------------------------------------ API

    @property
    def periods(self) -> np.ndarray:
        return 1.0 / self.frequencies

    def sigma_from_section(self, log10_res: np.ndarray, x_grid: np.ndarray,
                           depth_grid: np.ndarray) -> np.ndarray:
        """Nearest-neighbour map of a (n_z, n_x) section onto the mesh."""
        cc = self._m.active_cc
        ix = np.clip(np.searchsorted(x_grid, cc[:, 0]), 0, len(x_grid) - 1)
        iz = np.clip(np.searchsorted(depth_grid, -cc[:, 1]), 0, len(depth_grid) - 1)
        sigma = np.full(self._m.mesh.n_cells, _AIR_SIGMA)
        sigma[self._m.active_idx] = 10.0 ** (-log10_res[iz, ix])
        return sigma

    def response(self, section) -> tuple[np.ndarray, np.ndarray]:
        """Return (rho_a, phase_deg) of shape (n_freq, n_station).

        Phase is mapped to the first quadrant convention (0..90 deg) used by
        the 1D code and by EMTF products.
        """
        sigma = self.sigma_from_section(
            section.log10_res, section.x_grid, section.depth_grid
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self._sim.dpred(sigma)
        d = data.reshape(len(self.frequencies), 2, len(self.station_x))
        rho_a = d[:, 0, :]
        phase = d[:, 1, :] + 180.0  # SimPEG xy TE convention -> 0..90
        return rho_a, phase

    def response_tm(self, section) -> tuple[np.ndarray, np.ndarray]:
        """TM-mode (rho_a, phase_deg) of shape (n_freq, n_station).

        SimPEG's yx receiver already reports the phase in the 0..90 deg
        first-quadrant convention (validated against the 1D recursion).
        """
        sigma = self.sigma_from_section(
            section.log10_res, section.x_grid, section.depth_grid
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self._sim_tm.dpred(sigma)
        d = data.reshape(len(self.frequencies), 2, len(self.station_x))
        return d[:, 0, :], d[:, 1, :]

    def response_modes(self, section) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Both modes in one call: ``{"te": (rho, ph), "tm": (rho, ph)}``."""
        return {"te": self.response(section), "tm": self.response_tm(section)}
