# Checking the masks

The automatic filters left 426 candidate masks. This check covers all 40 candidates labeled cat, dog, bottle, cup, or chair in that output. Each mask was compared with its source-photo crop using Codex; the decisions below are AI-assisted and need human confirmation before training.

| Category | Retain | Reject | Unsure | Total checked |
| --- | --- | --- | --- | --- |
| Cat | 1 | 0 | 2 | 3 |
| Dog | 2 | 1 | 4 | 7 |
| Bottle | 3 | 0 | 2 | 5 |
| Cup | 3 | 3 | 0 | 6 |
| Chair | 0 | 19 | 0 | 19 |
| **Total** | **9** | **23** | **8** | **40** |

Retained masks had no obvious external occlusion in this check. Rejected masks showed missing outlines caused by other objects. Uncertain cases need a closer look, usually because the source is small or the boundary is hard to follow. The original category labels were kept.

The clearest problem was already-hidden objects. A dog behind a car window passed the automatic size and image-edge checks, even though most of its body was missing. Many chair masks contained only a backrest. Complete outlines also vary with pose: the lying dog was retained rather than rejected for looking different from a standing dog.

Small bottles raised a different issue. Their polygon outlines can be too coarse to judge confidently. Cup handles also showed why converted polygon labels need care: the outer silhouette may look usable while a handle hole has been lost. No holes were filled or contours repaired here.

These five categories were chosen for this check, not as the final project classes. Several chairs come from one photograph of overlapping rows, so the counts should not be read as COCO-wide acceptance rates.

![The 40 masks and their screening decisions](mask_review.png)

## Files and rerunning

`mask_review.csv` records the reason for each decision. Instance IDs combine the source-image ID with the zero-based line number in its polygon-label file.

After running the pilot, generate the comparison sheets with:

```sh
python build_review_sheet.py --data-dir /path/to/coco128-seg
```

Four source-photo comparison sheets are saved under `output/silhouette_pilot/`; they stay local. The script also regenerates the mask-only image above. Use `--sources-only` to inspect the source sheets before adding or revising CSV decisions.

Data: [Ultralytics COCO128-Seg](https://docs.ultralytics.com/datasets/segment/coco128-seg/), derived from [COCO](https://cocodataset.org/). This check uses converted polygon labels. The planned full dataset will use COCO 2017 instance annotations.
