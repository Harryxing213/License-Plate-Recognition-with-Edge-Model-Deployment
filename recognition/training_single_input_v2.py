#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_single_input_v2 import PlateRecognizerSingleInputV2, load_cnn_from_checkpoint
from plate_dataset import PlateDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="processed/train")
    parser.add_argument("--valid_path", type=str, default="processed/valid")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--init_checkpoint", type=str, default="data/model_batchnorm.pth")
    parser.add_argument("--output_checkpoint", type=str, default="data/checkpoints/model_single_input_v2.pth")
    parser.add_argument("--results_csv", type=str, default="data/checkpoints/results_single_input_v2.csv")
    return parser.parse_args()


def run_epoch(loader, model, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    plate_correct = 0
    plate_total = 0

    for images, _, targets in tqdm(loader):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits.flatten(0, 1), targets.flatten(0))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=-1)
        mask = targets != 0
        correct += ((preds == targets) & mask).sum().item()
        total += mask.sum().item()
        plate_correct += (((preds == targets) | ~mask).all(dim=1)).sum().item()
        plate_total += targets.size(0)
        total_loss += loss.item()

    return total_loss / len(loader), correct / total, plate_correct / plate_total


def main(args):
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    for name in ("train_path", "valid_path", "init_checkpoint", "output_checkpoint", "results_csv"):
        setattr(args, name, root / getattr(args, name))
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        PlateDataset(str(args.train_path)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        PlateDataset(str(args.valid_path)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = PlateRecognizerSingleInputV2()
    checkpoint = torch.load(root / args.init_checkpoint, map_location="cpu")
    load_cnn_from_checkpoint(model, checkpoint)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_plate_acc = -1.0
    rows = []

    for epoch in range(args.epochs):
        train_loss, train_acc, train_plate_acc = run_epoch(train_loader, model, criterion, device, optimizer)
        with torch.no_grad():
            valid_loss, valid_acc, valid_plate_acc = run_epoch(valid_loader, model, criterion, device)
        rows.append([epoch + 1, train_loss, train_acc, train_plate_acc, valid_loss, valid_acc, valid_plate_acc])
        print(
            f"Epoch {epoch + 1}/{args.epochs} train_acc={train_acc:.4f} train_plate={train_plate_acc:.4f} "
            f"valid_acc={valid_acc:.4f} valid_plate={valid_plate_acc:.4f}",
            flush=True,
        )
        if valid_plate_acc > best_plate_acc:
            best_plate_acc = valid_plate_acc
            torch.save(
                {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch + 1,
                    "valid_acc": valid_acc,
                    "valid_plate_acc": valid_plate_acc,
                },
                root / args.output_checkpoint,
            )
            print(f"Saved best single-input v2 checkpoint: {root / args.output_checkpoint}", flush=True)

    with open(root / args.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_plate_acc", "valid_loss", "valid_acc", "valid_plate_acc"])
        writer.writerows(rows)


if __name__ == "__main__":
    main(parse_args())
