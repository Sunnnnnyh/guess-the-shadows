# Guess the Shadows

**Single and Overlapping Silhouette Recognition — APS360 individual project**

Can a model recognize an object from its silhouette? What happens when two silhouettes overlap and parts of their outlines disappear? This project looks at both cases, using single objects as a reference for measuring how recognition changes with overlap.

## Current status

The project is at the proposal stage. So far, I have tested mask extraction, basic filtering, and overlapping silhouette generation on COCO128-Seg. Model training comes next.

The script processed 128 photographs containing 929 annotated objects. Of these, 426 passed the automatic size and image-boundary checks. They still need manual review, since these checks cannot tell whether an outline is complete. I selected a few examples for the preview.

## Dataset plan

I plan to use COCO 2017 instance masks and select ten everyday-object categories, aiming for 300–500 usable instances per category. The final selection will depend on mask quality. Screening will exclude crowd annotations, very small objects, substantial occlusion or truncation, and annotation errors.

Original photographs will be split into training, validation, and test sets in a 70:15:15 ratio with seed 360, before any overlapping images are made. All objects and generated samples from a photograph will stay in the same split to prevent data leakage.

Each input will be a black silhouette on a white 128 × 128 canvas. For overlapping inputs, two masks from different categories will be joined by pixelwise union, so the internal boundary disappears. Neither object's foreground area will exceed twice the other's.

Overlap will be measured using **IoS = intersection area / min(area A, area B)**:

| Level | IoS range |
| --- | --- |
| Low | 10–20% |
| Medium | 20–35% |
| High | 35–50% |

Each selected pair will have all three versions, with object scales held fixed and neither object clipped by the canvas. Labels, source IDs, transformations, and measured overlap will be saved with the generated samples.

The current script is an early trial. It records overlap as a fraction of each object's area and makes a few previews. The ten-category dataset and matched low/medium/high sets are still to be built.

## Models

I will start with a PyTorch CNN trained from scratch: three convolutional blocks with 32, 64, and 128 channels, each using a 3 × 3 convolution, ReLU, and 2 × 2 max pooling. Global average pooling and a linear layer will produce one logit per category. Binary cross-entropy with logits will be used for training; sigmoid scores and a threshold will determine the predicted labels.

The baseline will be one-vs-rest logistic regression on flattened silhouette pixels, with L2 regularization. Model selection, regularization strength, and prediction thresholds will use the validation set.

Both models will use the same data splits. Final test results will include exact-match accuracy and macro F1, reported separately for single objects and each overlap level. Neither model has been trained yet.

## Run the pilot

Use Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
```

Download and extract [COCO128-Seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip). Run:

```sh
python prepare_silhouette_pilot.py --data-dir /path/to/coco128-seg
```

The extracted directory should contain `images/train2017` and `labels/train2017`. The script saves masks, a preview sheet, class counts, and source records under `output/silhouette_pilot/`. This output folder is excluded from Git.

## Files

- `prepare_silhouette_pilot.py`: extracts masks, applies basic filters, and generates silhouette previews.
- `requirements.txt`: NumPy and Pillow, needed to run the script. Training dependencies will be added with the model code.
- `pilot_summary.json`: counts and notes from the trial run.
- `.gitignore`: keeps downloaded data, generated output, and model files out of Git.

## Data notes

COCO masks cover the visible part of an object, so some outlines are already incomplete before overlap is added. Converting polygon annotations to masks can also change fine contours and holes. Selecting complete outlines may favor clear, common poses. Results from these synthetic pairs will apply to the tested conditions; performance on real scenes will need separate evaluation.

Download the dataset separately using the link above. The images and annotations remain subject to their original terms.

- [COCO official website](https://cocodataset.org/)
- [Microsoft COCO: Common Objects in Context (ECCV 2014)](https://doi.org/10.1007/978-3-319-10602-1_48)
- [Official COCO annotation API](https://github.com/cocodataset/cocoapi)
- [Ultralytics COCO128-Seg documentation](https://docs.ultralytics.com/datasets/segment/coco128-seg/)
