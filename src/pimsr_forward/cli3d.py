"""Resumable 3D MT dataset builder with optional process parallelism."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .dataset3d import validate_sample3d


def _generate_one(params: dict, index: int) -> str:
    """Build one sample in a fresh worker; SimPEG leaks memory across solves."""
    from pimsr_geogen.volume3d import VolumeGenerator

    from .dataset3d import write_sample3d
    from .mt3d import MT3DForward

    frequencies = np.geomspace(params["fmin"], params["fmax"], params["nfreq"])
    stations = np.linspace(-0.35 * params["extent"], 0.35 * params["extent"], params["nstations"])
    generator = VolumeGenerator(
        seed=params["seed"],
        x_grid=np.linspace(-params["extent"] / 2, params["extent"] / 2, params["nx"]),
        y_grid=np.linspace(-params["extent"] / 2, params["extent"] / 2, params["ny"]),
        depth_grid=np.geomspace(10.0, params["depth"], params["nz"]),
    )
    forward = MT3DForward(frequencies, stations, stations, cell_size=params["cell_size"])
    volume = generator.sample(index)
    response = forward.response(volume)
    path = write_sample3d(
        Path(params["out"]) / f"sample_{index:07d}.h5",
        volume, response, frequencies, stations, stations,
        provenance={"generator_seed": params["seed"], "sample_index": index},
    )
    return str(path)


def build(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = 0
    for index in range(args.start, args.start + args.count):
        path = out / f"sample_{index:07d}.h5"
        if path.exists():
            try:
                validate_sample3d(path)
                skipped += 1
                continue
            except (OSError, ValueError, KeyError):
                path.unlink()
        pending.append(index)

    params = {
        "out": str(out), "seed": args.seed, "nx": args.nx, "ny": args.ny, "nz": args.nz,
        "extent": args.extent, "depth": args.depth, "cell_size": args.cell_size,
        "nstations": args.nstations, "nfreq": args.nfreq, "fmin": args.fmin, "fmax": args.fmax,
    }
    if args.workers <= 1:
        for index in pending:
            _generate_one(params, index)
    else:
        # spawn + one task per child keeps solver memory leaks bounded.
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=get_context("spawn"), max_tasks_per_child=1
        ) as pool:
            for _ in pool.map(_generate_one, (params,) * len(pending), pending):
                pass
    return {"completed": len(pending), "skipped": skipped, "output": str(out.resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--extent", type=float, default=30000.0)
    parser.add_argument("--depth", type=float, default=30000.0)
    parser.add_argument("--cell-size", type=float, default=2500.0)
    parser.add_argument("--nstations", type=int, default=8)
    parser.add_argument("--nfreq", type=int, default=12)
    parser.add_argument("--fmin", type=float, default=0.01)
    parser.add_argument("--fmax", type=float, default=100.0)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
