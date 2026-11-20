# Experiment History

## Detection

The detector uses YOLOv8n and a one-class license plate dataset. Training, validation, saved predictions and a Tkinter GUI are present in the internship files.

The final GUI points to the `train8` best checkpoint. The older command-line pipeline pointed to `train3`; both now use `detection/weights/best.pt`.

The saved `train8` configuration records 50 epochs, batch 16, image size 640 and initialization from `train6/weights/last.pt`. The short original README says 250 epochs; the saved per-run configuration is retained to avoid treating that summary as the duration of `train8` itself.

The highest mAP50 in the training CSV is 0.995. The supplied best checkpoint records mAP50 0.99391 and mAP50-95 0.80232. These refer to different selection criteria/epochs. The main README uses the checkpoint values.

The original large Roboflow export and its mixed class-name YAML are separate from the one-class `train_exp/data.yaml` used by the 519-image experiment. Only the final one-class configuration is included here.

## Recognition and Deployment

| Route | Work done | Recorded outcome |
| --- | --- | --- |
| CNN + Transformer decoder | CNN feature extraction and autoregressive character prediction | LayerNorm/exNorm and decoder-input deployment problems on the tested RV1106 runtime |
| LayerNorm graph rewrites | Affine approximations, calibrated rewrites and INT32/Cast attempts | Approximation reduced accuracy; unsupported Cast/runtime input issues remained |
| BatchNorm decoder | Replace decoder normalization and fine-tune | Multi-input / integer-token deployment issues remained |
| Single-input v1 | Remove decoder-token input | Deployed, but plate accuracy was too low |
| Single-input v2 | CNN + residual horizontal Conv1D blocks | Final route, validation plate accuracy 0.9644432194046306 |

The original Transformer test report records 32,575 samples, autoregressive character accuracy 0.9885 and full-plate accuracy 0.9554. It belongs to the earlier route and is not used as a result for the final v2 model.

The final model uses 7 plate characters plus EOS. Its 70-class vocabulary also contains PAD and BOS. It is not trained here as a variable-length or 8-character new-energy plate recognizer.

## Internship Notes

The camera notes cover UART, Linux camera nodes and logs, I2C sensor configuration, MIPI CSI, ISP, UVC, ADB, RTSP and USB Gadget. They explain the surrounding camera bring-up and debugging work. They are not evidence that these components form a completed real-time recognition pipeline in the final code.

The OpenCV practice files cover grayscale transforms, spatial filters, Fourier transforms and morphology. They remain learning material in the original folder and are not dependencies of the final project.

## Code Organization

- Keep the final recognizer architecture, vocabulary order and supplied model weights.
- Keep the original CNN class and parameter names; remove the unused Transformer definitions from the final CNN dependency file.
- Keep training initialization from the BatchNorm checkpoint.
- Align the detector GUI and command-line default checkpoint.
- Resolve recognition training paths relative to its script directory.
- Save newly trained checkpoints and newly exported models separately from supplied artifacts.
- Add direct cropped-image prediction and an evaluation entry point using the existing metric implementation.
- Keep only the final W8A8 conversion route; remove the old misleading `none`/FP16 switches from the final entry point.
- Retain board C sources and the original RV1106 test binaries. Use SDK-provided headers and runtime libraries when rebuilding.
