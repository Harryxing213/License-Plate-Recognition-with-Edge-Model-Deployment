#!/usr/bin/env python3
from pathlib import Path
import argparse

import torch

from model_single_input_v2 import PlateRecognizerSingleInputV2


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/model_single_input_v2.pth")
    parser.add_argument("--output", default="data/exports/plate_recognizer_single_input_v2.onnx")
    args = parser.parse_args()
    checkpoint_path = root / args.checkpoint
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = PlateRecognizerSingleInputV2()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image = torch.randn(1, 3, 64, 256, dtype=torch.float32)
    torch.onnx.export(
        model,
        image,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
        external_data=False,
    )
    print(f"Exported {output_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Valid char acc: {checkpoint.get('valid_acc')}")
    print(f"Valid plate acc: {checkpoint.get('valid_plate_acc')}")


if __name__ == "__main__":
    main()
