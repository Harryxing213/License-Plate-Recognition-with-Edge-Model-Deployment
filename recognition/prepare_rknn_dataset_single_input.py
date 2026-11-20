#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument("--src", default="processed/train/images")
    parser.add_argument("--out", default="data/rknn_dataset_single_input.txt")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be > 0")

    root = Path(__file__).resolve().parent
    src = root / args.src
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )[: args.count]

    if not images:
        raise ValueError(f"No calibration images found: {src}")

    with out.open("w", encoding="utf-8") as f:
        for path in images:
            f.write(str(path.resolve()) + "\n")

    print(f"Wrote {len(images)} image paths to {out}")


if __name__ == "__main__":
    main()
