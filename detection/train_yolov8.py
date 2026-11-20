from yolov8_pipeline import DATA_YAML, DEFAULT_MODEL, train_model


def main():
    train_model(
        data_path=DATA_YAML,
        model_path=DEFAULT_MODEL,
        epochs=50,
        imgsz=640,
        batch=4,
    )


if __name__ == "__main__":
    main()
