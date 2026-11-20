from pathlib import Path
import argparse

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "train_exp" / "data.yaml"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRAINED = ROOT / "weights" / "best.pt"


def train_model(data_path: Path, model_path: str, epochs: int, imgsz: int, batch: int):
    model = YOLO(model_path)
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "runs" / "detect"),
        name="train",
    )


def validate_model(model_path: Path, data_path: Path, split: str):
    model = YOLO(str(model_path))
    model.val(
        data=str(data_path),
        split=split,
        project=str(ROOT / "runs" / "detect"),
    )


def predict_images(model_path: Path, source: str, conf: float):
    model = YOLO(str(model_path))
    model.predict(
        source=source,
        conf=conf,
        save=True,
        project=str(ROOT / "runs" / "detect"),
    )


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 training and evaluation pipeline")
    parser.add_argument(
        "mode",
        choices=["train", "val", "test", "predict"],
        help="train: train model, val: validate on valid split, test: evaluate on test split, predict: save predictions",
    )
    parser.add_argument("--data", default=str(DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint for training")
    parser.add_argument("--weights", default=str(DEFAULT_TRAINED), help="Trained weights for val/test/predict")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--source", default=str(ROOT / "train_exp" / "test" / "images"), help="Prediction source")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for prediction")

    args = parser.parse_args()

    data_path = Path(args.data)

    if args.mode == "train":
        train_model(data_path, args.model, args.epochs, args.imgsz, args.batch)
    elif args.mode == "val":
        validate_model(Path(args.weights), data_path, split="val")
    elif args.mode == "test":
        validate_model(Path(args.weights), data_path, split="test")
    elif args.mode == "predict":
        predict_images(Path(args.weights), args.source, args.conf)


if __name__ == "__main__":
    main()
