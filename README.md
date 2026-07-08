# pimsr-forward

Forward modeling engine for the PIMSR (Physics-Informed Multi-modal Subsurface Reconstruction) project.

Turns stochastic geology models from [`pimsr-geogen`](https://github.com/TheLitis/pimsr-geogen) into realistic geophysical observables:

- **Magnetotellurics (MT), 2D TE+TM** (`mt2d.py`) — SimPEG-based finite-volume
  forward for 2D sections: TE (`Simulation2DElectricField`) and TM
  (`Simulation2DMagneticField`) apparent resistivity + phase pseudo-sections
  at 16 stations x 24 periods. Validated against the 1D analytic response to
  0.4% (TE) / 0.13% (TM) median error on layered models.
- **Magnetotellurics (MT), 1D analytic** — exact impedance recursion for layered media (Wait recursion). Apparent resistivity + phase over a configurable period band.
- **Gravity** — vertical attraction of right rectangular prisms (Nagy 1966 closed form), evaluated on a surface profile.
- **Sensor / noise model** — realistic MT error floors (impedance-proportional +
  absolute floor), AR(1)-correlated galvanic distortion calibrated on real
  USArray residuals, per-station static shift, and **per-mode severity**: TM
  curves receive stronger per-section static shifts and distortion than TE,
  matching real yx-impedance behaviour.
- **Dataset builders** — 1D per-station (`dataset.py`) and 2D section
  (`dataset2d.py`, chunk-resumable, shard merging) HDF5 training sets for
  `pimsr-inversion`. The 2D builder writes 4-channel TE+TM observations.

## Physics validation

The 1D MT kernel is cross-checked against half-space and two-layer analytic limits in tests
(`tests/test_mt1d.py`), and against SimPEG's 1D MT simulation in
[`pimsr-benchmarks`](https://github.com/TheLitis/pimsr-benchmarks). The 2D
TE/TM forwards are validated against the 1D analytic response on layered
sections (`tests/test_mt2d.py`).

The gravity kernel is checked against the infinite-slab Bouguer limit and prism symmetry
identities (`tests/test_gravity.py`).

## Install

```bash
pip install -e .
```

## Usage

```python
from pimsr_forward import mt1d_response, prism_gz, SensorModel

rho_app, phase = mt1d_response(resistivities, thicknesses, periods)
```

Dataset generation CLI:

```bash
# 1D per-station dataset
pimsr-forward-dataset --geology geology.h5 --out dataset.h5 --seed 42
# 2D TE+TM section dataset (requires the mt2d extra: pip install -e ".[mt2d]")
pimsr-forward-dataset2d --n 1000 --seed 7 --out ds2d.h5
```

## License

MIT
