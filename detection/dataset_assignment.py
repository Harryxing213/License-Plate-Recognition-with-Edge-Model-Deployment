from pathlib import Path
import argparse
import random
import shutil


ROOT = Path(__file__).resolve().parent


def copy_samples(samples, images_dir, labels_dir):
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for img, label in samples:
        shutil.copy2(img, images_dir / img.name)
        shutil.copy2(label, labels_dir / label.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "train_exp")
    parser.add_argument("--out", type=Path, default=ROOT / "train_exp")
    args = parser.parse_args()
    src_images = args.src / "images"
    src_labels = args.src / "labels"
    out_root = args.out

    # 新建数据时沿用原来的 70% / 20% / 10% 划分。
    for split in ("train", "valid", "test"):
        if (out_root / split).exists():
            raise FileExistsError(f"Split already exists: {out_root / split}")

    samples = []
    for img in sorted(src_images.glob("*.jpg")):
        label = src_labels / f"{img.stem}.txt"
        if label.exists():
            samples.append((img, label))
    if len(samples) < 10:
        raise ValueError("At least 10 labeled .jpg images are required")

    random.Random(42).shuffle(samples)
    train_end = int(0.7 * len(samples))
    valid_end = int(0.9 * len(samples))
    splits = {
        "train": samples[:train_end],
        "valid": samples[train_end:valid_end],
        "test": samples[valid_end:],
    }
    for split, items in splits.items():
        copy_samples(items, out_root / split / "images", out_root / split / "labels")
        print(f"{split}: {len(items)}")

    (out_root / "data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['license_plate']\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
