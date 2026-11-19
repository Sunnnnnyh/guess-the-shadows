# COCO silhouette dataset: first build

The first build uses original COCO 2017 instance annotations, rather than the converted COCO128-Seg polygons used in the proposal pilot. The ten **candidate** classes are cat, dog, bird, bottle, cup, chair, car, airplane, umbrella, and teddy bear. Final class selection is still open.

The automatic pass extracted 400 masks per class and generated 3,000 different pairs, each at low, medium, and high overlap. An initial source-photo check then excluded 28 rejected or uncertain instances and every overlapping image made from them. The active manifests contain **3,972 single silhouettes and 8,871 overlapping silhouettes: 12,843 images**.

These are first-pass candidates. The source-photo check covered 40 of the 200 queued training examples; it did not establish that all remaining masks have complete outlines. Chairs were particularly problematic in that check. The 100-mask review from the proposal remains a separate, earlier record and is not counted again here.

## Processing rules

- Exclude crowd annotations, missing or invalid masks, foreground area below 500 source pixels, bounding-box short side below 24 pixels, and boxes within two pixels of the source-image boundary. Apply the area, size and boundary checks again after rasterization.
- Decode masks with `pycocotools`. Do not fill holes or repair missing body parts. Multiple polygons are combined according to the original COCO annotation.
- Sort all COCO train2017 image IDs, shuffle with seed 360, then assign 70%/15%/15% to training/validation/test. Every object and derivative of a source photograph stays in that split.
- Within each category and split, prefer annotations containing a single polygon, then order by a seeded SHA-256 key. This preference is a selection bias, not a guarantee of completeness. Initial per-class quotas are 280/60/60.
- Crop each single mask to its foreground box, resize proportionally to a longest side of 112 pixels, and center it on a white 128 × 128 canvas. Foreground is black. All resizing uses nearest-neighbor interpolation.
- Pair different categories from different photographs within the same split. Approximate equal foreground areas, then reduce both masks by one common factor if needed to fit the canvas. Reject a pair if its foreground-area ratio exceeds two.
- Keep the resulting object sizes fixed across all three overlap levels. Search integer translations for IoS bands `[0.10, 0.20)`, `[0.20, 0.35)`, and `[0.35, 0.50)`, where IoS is intersection divided by the smaller foreground area. Center the joint bounding box, without clipping either mask, and save the binary union.
- Keep source annotation IDs, image IDs, labels, final mask sizes, positions and measured overlap. Category pairs are sampled cyclically so that their counts initially differ by at most one within a split. Subsequent visual exclusions can change that balance.

## Current active split sizes

| Split | Single | Low | Medium | High | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,772 | 2,057 | 2,057 | 2,057 | 8,943 |
| Validation | 600 | 450 | 450 | 450 | 1,950 |
| Test | 600 | 450 | 450 | 450 | 1,950 |
| Total | 3,972 | 2,957 | 2,957 | 2,957 | 12,843 |

## Run

Use Python 3.10+ and install `requirements-data.txt` in your project environment.

```sh
python download_coco_annotations.py --data-dir ../data/coco2017
python prepare_dataset.py --annotations ../data/coco2017/annotations/instances_train2017.json --output output/formal_v1 --instances-per-class 400 --pairs 3000
python render_dataset_review.py --dataset output/formal_v1 --download
```

Record source-photo decisions in `output/formal_v1/review_queue.csv`, then run:

```sh
python apply_mask_reviews.py --dataset output/formal_v1
python validate_dataset.py --dataset output/formal_v1
python package_dataset.py --dataset output/formal_v1 --archive output/guess_the_shadows_candidates_v1.zip
```

The review queue contains 20 training examples from different source photographs per class. Use the numbered source-photo comparison sheets and `review_index.json` to locate them. `reject` and `uncertain` decisions remove that instance and all its derivatives from the training manifests; `pending` leaves an automatic candidate in the first-pass set. The existing initial decisions and counts are saved in `evidence/coco2017_first_pass/` so they can be copied back after rebuilding.

The three JSONL split files are the loading lists. Each row gives a PNG path, condition and `label_indices`; convert those indices into a ten-element multi-hot target. A pair at low/medium/high overlap has the same `pair_id`. Images excluded by review may still exist locally, so do not load the dataset by enumerating folders.

`category_scan.csv` records raw counts and automatic rejection reasons for all 80 COCO categories. `class_counts.csv` records active single-object counts. `screening_summary.json` gives counts after visual exclusions. `validation_results.json` checks labels, saved image contents, no source leakage, fixed pair sizes, no clipping, and actual IoS. It does not certify visual completeness.

The training archive contains active images, masks, labels and metadata. Source photographs and full annotations stay local and are excluded from Git.

## New data for final testing

The internal test split above is held out from model selection. A separate final check on newly collected photographs or an unused external dataset is still to be collected. A practical next step is a small set of separately photographed objects from the final selected classes, with masks checked before synthesis. It must stay out of training, threshold selection and category selection. No external-test result is claimed in this build.

Source: [COCO](https://cocodataset.org/) and its [official annotation API](https://github.com/cocodataset/cocoapi). The download script uses COCO's official image-storage bucket and records the annotation archive's SHA-256 checksum.
