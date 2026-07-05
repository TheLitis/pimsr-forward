"""CLI: build a 2D profile dataset (SimPEG TE forward)."""

from __future__ import annotations

import argparse

from .dataset2d import build_dataset_2d


def main() -> None:
    p = argparse.ArgumentParser(description="Build a 2D MT profile dataset")
    p.add_argument("--out", required=True, help="output dataset HDF5")
    p.add_argument("--n", type=int, required=True, help="number of sections")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0,
                   help="first section index (for sharded generation)")
    args = p.parse_args()

    build_dataset_2d(args.out, args.n, seed=args.seed, start_index=args.start_index)
    print(f"wrote {args.n} sections -> {args.out}")


if __name__ == "__main__":
    main()
