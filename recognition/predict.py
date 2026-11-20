from pathlib import Path
import argparse

import torch

from model_single_input_v2 import PlateRecognizerSingleInputV2
from plate_dataset import PlateDataset, VOCAB


ROOT = Path(__file__).resolve().parent


def load_model(checkpoint_path):
    model = PlateRecognizerSingleInputV2()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def decode_plate(indices):
    chars = []
    for idx in indices:
        if idx == 2:  # EOS
            break
        if idx not in (0, 1):
            chars.append(VOCAB[idx])
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Cropped plate image or folder")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/model_single_input_v2.pth")
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    source = args.source.expanduser().resolve()
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        images = [source]
    if not images:
        raise ValueError(f"No images found: {source}")

    # 与训练保持相同的 RGB、resize 和 /255 预处理。
    with torch.no_grad():
        for path in images:
            image = torch.from_numpy(PlateDataset.image_process(str(path))).unsqueeze(0)
            indices = model(image).argmax(dim=-1)[0].tolist()
            print(f"{path.name}\t{decode_plate(indices)}")


if __name__ == "__main__":
    main()
