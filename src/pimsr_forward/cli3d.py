"""Small resumable 3D MT dataset builder; production orchestration stays external."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pimsr_geogen.volume3d import VolumeGenerator

from .dataset3d import validate_sample3d, write_sample3d
from .mt3d import MT3DForward


def build(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frequencies = np.geomspace(args.fmin, args.fmax, args.nfreq)
    stations = np.linspace(-0.35 * args.extent, 0.35 * args.extent, args.nstations)
    generator = VolumeGenerator(
        seed=args.seed,
        x_grid=np.linspace(-args.extent / 2, args.extent / 2, args.nx),
        y_grid=np.linspace(-args.extent / 2, args.extent / 2, args.ny),
        depth_grid=np.geomspace(10.0, args.depth, args.nz),
    )
    forward = MT3DForward(frequencies, stations, stations, cell_size=args.cell_size)
    completed = skipped = 0
    for index in range(args.start, args.start + args.count):
        path = out / f"sample_{index:07d}.h5"
        if path.exists():
            try:
                validate_sample3d(path)
                skipped += 1
                continue
            except (OSError, ValueError, KeyError):
                path.unlink()
        volume = generator.sample(index)
        response = forward.response(volume)
        write_sample3d(
            path, volume, response, frequencies,
            provenance={"generator_seed": args.seed, "sample_index": index},
        )
        completed += 1
    return {"completed": completed, "skipped": skipped, "output": str(out.resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
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
