from __future__ import annotations

import argparse
from pathlib import Path

from sprinkler_hd_gan.synthetic_pairs import write_synthetic_dataset
from sprinkler_hd_gan.util import load_yaml_config


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic paired dataset for training demos.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--train", type=int, default=400)
    p.add_argument("--val", type=int, default=40)
    args = p.parse_args()

    cfg = load_yaml_config(args.config)
    c = cfg["canvas"]
    write_synthetic_dataset(
        args.out,
        args.train,
        args.val,
        width=int(c["width"]),
        height=int(c["height"]),
        mm_per_pixel=float(c["mm_per_pixel"]),
    )


if __name__ == "__main__":
    main()
