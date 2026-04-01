# SAM3 Tennis ROI Tracking

Modal-based video segmentation pipeline for ROI tracking in tennis broadcast streams using SAM3.

## What It Does

- Runs SAM3 video segmentation on a tennis video in Modal.
- Annotates the detected target, currently prompted with `"logo"` by default.
- Saves the processed video and metrics JSON to a Modal Volume.
