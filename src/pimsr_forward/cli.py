"""CLI: forward-model a geology batch into a training dataset."""

from __future__ import annotations

import argparse

from .dataset import build_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Build (observables, targets) dataset")
    p.add_argument("--geology", required=True, help="geology HDF5 from pimsr-geogen")
    p.add_argument("--out", required=True, help="output dataset HDF5")
    p.add_argument("--seed", type=int, default=0, help="noise RNG seed")
    p.add_argument("--n-periods", type=int, default=24)
    args = p.parse_args()

    n = build_dataset(args.geology, args.out, seed=args.seed, n_periods=args.n_periods)
    print(f"wrote {n} samples -> {args.out}")


if __name__ == "__main__":
    main()
