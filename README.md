# pimsr-forward

Forward modeling engine for the PIMSR (Physics-Informed Multi-modal Subsurface Reconstruction) project.

Turns stochastic geology models from [`pimsr-geogen`](https://github.com/TheLitis/pimsr-geogen) into realistic geophysical observables:

- **Magnetotellurics (MT), 1D analytic** — exact impedance recursion for layered media (Wait recursion). Apparent resistivity + phase over a configurable period band.
- **Gravity** — vertical attraction of right rectangular prisms (Nagy 1966 closed form), evaluated on a surface profile.
- **Sensor / noise model** — realistic MT error floors (impedance-proportional + absolute floor), gravimeter drift + white noise, per-station static shift.
- **Dataset builder** — batches geology models into HDF5 training sets `(observables, targets)` for `pimsr-inversion`.

## Physics validation

The MT kernel is cross-checked against half-space and two-layer analytic limits in tests
(`tests/test_mt1d.py`), and against SimPEG's 1D MT simulation in
[`pimsr-benchmarks`](https://github.com/TheLitis/pimsr-benchmarks).

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
pimsr-forward-dataset --geology geology.h5 --out dataset.h5 --seed 42
```

## License

MIT
