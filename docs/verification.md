# Verification

The prepared code was checked in a temporary Python 3.12 environment on macOS. The original internship folder was read only.

## Completed

- Parsed every Python source file.
- Loaded the final recognition checkpoint with strict parameter matching after organizing the CNN dependency.
- Ran the first 20 images in the original recognition test split: 20 / 20 correct with the PyTorch model.
- Checked the supplied ONNX graph and compared its outputs with PyTorch on the same 20 images. Predicted token indices all match; maximum absolute logit difference: 1.5258789e-05.
- Re-exported the supplied checkpoint with the organized export script.
- Regenerated all 20 recorded board inputs: byte-for-byte agreement with the original raw files using OpenCV 4.13.0.92.
- Ran the supplied YOLO checkpoint on a test image successfully.
- Ran the prediction and evaluation command-line programs.
- Ran one training epoch with 2 training and 2 validation images, including checkpoint and CSV writing. Temporary outputs were written outside the prepared project.
- Checked dataset splitting on 20 synthetic file pairs: 14 / 4 / 2, no overlap, and no overwrite of existing splits.
- Checked both C sources with the host compiler against the original SDK header. This is a syntax check, not a target link or board execution.

The 2-image training run checks the code path and file writing. It does not reproduce training accuracy. The 20-image recognition check does not replace full-split evaluation.

## Recorded Results

The final recognition checkpoint records epoch 40, validation token accuracy 0.993901598676957 and full-plate accuracy 0.9644432194046306. Both metrics include EOS.

The saved YOLO best checkpoint records validation mAP50 0.99391 and mAP50-95 0.80232.

RV1106 average run-only latency 3.129 ms and the 20 / 20 board sample result come from the existing final-package README. No raw benchmark log or per-sample board prediction log was found alongside that record. These hardware results were not rerun during this preparation.

RKNN conversion was not rerun on this macOS machine. Use the matching Linux Toolkit2 and RV1106 runtime/SDK to convert or rebuild for hardware.

## Preprocessing Versions

OpenCV 5.1 produced pixel differences of at most one intensity level in the first inspected board samples. OpenCV 4.12.0.88 and 4.13.0.92 both reproduced all 20 original raw files exactly. The project pins OpenCV 4.13.0.92 to keep this preprocessing consistent.

`requirements_verified.txt` records the exact top-level versions used in the final local verification; it is not a record of the original training environment.

## Source Data Gaps

- The detection test directory contains 61 images and 50 label files. The missing files need review before reporting annotated test-set metrics.
- The final recognition training CSV contains only its header. The checkpoint provides the recorded validation metrics.
- A usable original CCPD cropping / perspective-correction script was not found in the current files. The final recognition programs expect already cropped plate images; no reconstructed preprocessing script is attributed to the original project.
- The camera knowledge document describes interfaces and debugging ideas; those notes do not establish a completed live camera pipeline.

The detailed local check results are saved in `metrics/verification.json`.
