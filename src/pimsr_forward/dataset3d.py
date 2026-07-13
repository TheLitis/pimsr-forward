"""Atomic, resumable HDF5 contract for 3D MT samples."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

__all__ = ["write_sample3d", "validate_sample3d"]

SCHEMA_VERSION = 1


def write_sample3d(path, volume, response, frequencies, provenance=None) -> Path:
    """Write one sample via ``.part`` then atomically publish it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with h5py.File(part, "w") as f:
        f.attrs["schema"] = "pimsr-3d"
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["provenance"] = json.dumps(provenance or {}, sort_keys=True)
        f.attrs["scenario"] = volume.scenario
        f.attrs["seed"] = volume.seed
        f.attrs["wall_time_s"] = response.wall_time_s
        f.create_dataset("observations/apparent_resistivity", data=response.apparent_resistivity.astype("f4"))
        f.create_dataset("observations/phase", data=response.phase.astype("f4"))
        f.create_dataset("target/log10_resistivity", data=volume.log10_res.astype("f4"), compression="gzip")
        f.create_dataset("coordinates/frequencies", data=np.asarray(frequencies))
        f.create_dataset("coordinates/x", data=volume.x_grid)
        f.create_dataset("coordinates/y", data=volume.y_grid)
        f.create_dataset("coordinates/depth", data=volume.depth_grid)
        f.create_dataset("coordinates/modes", data=np.asarray(response.modes, dtype="S2"))
    part.replace(path)
    validate_sample3d(path)
    return path


def validate_sample3d(path) -> None:
    with h5py.File(path) as f:
        if f.attrs.get("schema") != "pimsr-3d" or f.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported PIMSR 3D schema")
        obs = f["observations/apparent_resistivity"]
        phase = f["observations/phase"]
        target = f["target/log10_resistivity"]
        expected_obs = (
            len(f["coordinates/frequencies"]), len(f["coordinates/modes"]),
            len(f["coordinates/y"]), len(f["coordinates/x"]),
        )
        expected_target = (len(f["coordinates/depth"]), len(f["coordinates/y"]), len(f["coordinates/x"]))
        if obs.shape != expected_obs or phase.shape != expected_obs or target.shape != expected_target:
            raise ValueError("3D dataset dimensions do not match coordinates")
        if not np.isfinite(obs[:]).all() or not np.isfinite(phase[:]).all() or not np.isfinite(target[:]).all():
            raise ValueError("3D sample contains non-finite values")
