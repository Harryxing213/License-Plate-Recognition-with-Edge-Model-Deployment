import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="plate_recognizer_single_input_v2.onnx")
    parser.add_argument("--dataset", default="rknn_dataset_single_input.txt")
    parser.add_argument("--output", default="exports/plate_recognizer_single_input_v2_uint8_w8a8.rknn")
    parser.add_argument("--optimization_level", type=int, default=3)
    args = parser.parse_args()

    # RKNN Toolkit2 在单独的 Linux 环境中运行。
    from rknn.api import RKNN

    root = Path(__file__).resolve().parent / "data"
    onnx_path = root / args.onnx
    dataset_path = root / args.dataset
    output_path = root / args.output
    for path in (onnx_path, dataset_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(
            target_platform="rv1106",
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            quant_img_RGB2BGR=False,
            optimization_level=args.optimization_level,
            quantized_dtype="w8a8",
            quantized_algorithm="normal",
            quantized_method="channel",
        )
        if ret not in (0, None):
            raise RuntimeError(f"config failed: {ret}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"export_rknn failed: {ret}")
        print(f"Exported {output_path}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
