# Guess the Shadows

**Single and Overlapping Silhouette Recognition — APS360 individual project**

Given a binary silhouette containing one object or two overlapping objects, predict the categories present. The project studies recognition from shape alone and how performance changes as overlap increases.

## Current status

This repository contains proposal-stage work. A small COCO128-Seg pilot checks mask extraction, basic screening, and silhouette composition. The CNN and baseline have **not** been trained, and there are no classification results yet.

The pilot processes 128 source photographs and 929 polygon instances; 426 instances pass its automatic size and image-boundary checks. These counts do not establish that the masks are complete or suitable for final training. Manual review is still required. The preview uses a small, manually selected set of examples.

## Planned dataset

- Source: COCO 2017 instance-segmentation annotations.
- Target: ten everyday-object categories, approximately 300–500 usable instances per category. Final classes and counts depend on screening.
- Inspect candidate masks and exclude crowd annotations, very small objects, substantial truncation or occlusion, and annotation errors.
- Split by original photograph before synthesis, keeping every instance and derivative of a photograph in the same partition. Proposed split: 70% training, 15% validation, 15% test; seed 360.
- Render foreground masks as black silhouettes on white 128 × 128 canvases.
- Combine masks from different categories by their binary union. Keep foreground areas within a factor of two.
- Measure overlap as **IoS = intersection area / min(area A, area B)**. Proposed bins: low 10–20%, medium 20–35%, high 35–50%.
- Keep object scales fixed across the three overlap versions of each selected pair, and avoid clipping.
- Retain source IDs, labels, transforms, and overlap measurements.

The existing pilot predates the final proposal settings. It measures intersection fractions separately for both instances and generates a few previews; it does **not** yet implement the complete ten-category dataset or matched low/medium/high evaluation sets.

## Planned models and evaluation

The initial PyTorch CNN will use three convolutional blocks with 32, 64, and 128 channels, 3 × 3 convolutions, ReLU, and 2 × 2 max pooling. Global average pooling and a linear layer will produce one logit per category. Training will use binary cross-entropy with logits, with sigmoid scores thresholded at inference.

The baseline will be one-vs-rest logistic regression on flattened silhouette pixels with L2 regularization. Model choices, regularization strength, and prediction thresholds will use validation data only.

Both models will use the same source partitions and be evaluated with exact-match accuracy and macro F1, separately for single objects and each overlap level.

## Run the pilot

Use Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
```

Download and extract [COCO128-Seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip). Run:

```sh
python prepare_silhouette_pilot.py --data-dir /path/to/coco128-seg
```

The extracted directory should contain `images/train2017` and `labels/train2017`. Results are written under `output/silhouette_pilot/`, including individual masks, a preview sheet, class counts, and provenance metadata. Generated data are excluded from Git.

## Files

- `prepare_silhouette_pilot.py`: initial mask screening and preview-composition script.
- `requirements.txt`: dependencies for the pilot; model-training dependencies will be documented with the implementation.
- `pilot_summary.json`: aggregate counts from the local pilot run.

## Limitations and data attribution

COCO masks describe visible regions, so a mask can already be incomplete before synthetic overlap is added. Polygon conversion may also change fine contours and holes. Screening for complete outlines can introduce selection bias, and synthetic overlaps do not establish robustness to real-world occlusion.

Source photographs, masks, course handouts, and personal student details are not distributed in this repository. Dataset images and annotations retain their respective terms; cite their original sources when using them.

- [COCO official website](https://cocodataset.org/)
- [Microsoft COCO: Common Objects in Context (ECCV 2014)](https://doi.org/10.1007/978-3-319-10602-1_48)
- [Official COCO annotation API](https://github.com/cocodataset/cocoapi)
- [Ultralytics COCO128-Seg documentation](https://docs.ultralytics.com/datasets/segment/coco128-seg/)

## Next steps

- Select final categories and document manual screening.
- Implement matched overlap levels and record final dataset statistics.
- Implement the CNN and logistic-regression baseline.
- Train, select settings on validation data, and evaluate on held-out data.
- Analyze category-level errors and ambiguous silhouette pairs.
