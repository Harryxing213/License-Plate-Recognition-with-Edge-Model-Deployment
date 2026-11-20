#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def load_labels(label_path):
    rows = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_name, label = line.split("\t")
            rows.append((image_name, label))
    return rows


def image_to_rknn_int8_raw(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (256, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    quantized = img.astype(np.int16) - 128
    return quantized.astype(np.int8)


def main():
    parser = ArgumentParser()
    parser.add_argument("--src", default="processed/test")
    parser.add_argument("--out", default="data/board_test_v2_raw")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be > 0")

    root = Path(__file__).resolve().parent
    src = root / args.src
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(src / "label.txt")[: args.count]
    manifest_path = out / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx, (image_name, label) in enumerate(labels):
            image_path = src / "images" / image_name
            raw = image_to_rknn_int8_raw(image_path)
            raw_name = f"{idx:04d}_{Path(image_name).stem}.raw"
            raw.tofile(out / raw_name)
            manifest.write(f"{idx}\t{image_name}\t{raw_name}\t{label}\n")
    print(f"Wrote {len(labels)} raws to {out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
