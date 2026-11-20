from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from plate_dataset import PlateDataset
from predict import ROOT, load_model
from training_single_input_v2 import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=Path, default=ROOT / "processed/test")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/model_single_input_v2.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0: evaluate the whole split")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    dataset = PlateDataset(args.test_path)
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = load_model(args.checkpoint)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        loss, char_acc, plate_acc = run_epoch(loader, model, criterion, torch.device("cpu"))
    print(f"Total samples: {len(dataset)}")
    print(f"Loss: {loss:.6f}")
    print(f"Character accuracy (including EOS): {char_acc:.6f}")
    print(f"Full-plate accuracy (including EOS): {plate_acc:.6f}")


if __name__ == "__main__":
    main()
