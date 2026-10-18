# Mask screening trial

This trial checks whether automatic size and boundary filters are enough to select complete object outlines. They are not: an object can sit inside the photograph while another object hides most of it.

## Sample and review

After automatic filtering, the sample was drawn with Python's `random.Random(360)` from sorted instance IDs: three cats, four dogs, four bottles, four cups, and five chairs. This gives 20 masks from five categories. Categories were chosen for this inspection, not sampled to represent all of COCO. Some instances share a source photograph.

Codex compared each mask with its source-image crop and recorded an **AI-assisted visual assessment**, not a human annotation. The decisions are provisional:

| Decision | Count | Meaning |
| --- | --- | --- |
| Retain | 7 | No obvious external occlusion found in this inspection; candidate for further checking. |
| Reject | 8 | Visible occlusion leaves only part of the object's outline. |
| Uncertain | 5 | Contact with another object, clutter, or coarse polygons needs closer review. |

These counts describe this small sample only. They do not estimate dataset-wide acceptance rates or establish that a category is suitable for the final project. The five categories are not the final class selection.

![Sampled masks and provisional screening decisions](mask_review.png)

`mask_review.csv` gives the instance ID, original COCO category, decision, reason, and reviewer for every sample. The ID consists of the source-image ID followed by the zero-based polygon line index. Category labels have not been changed.

## What the inspection found

- The dog in instance `000000000400_00` is mostly hidden by a car window. Its mask passed the automatic checks because the visible region is large enough and does not touch the image edge.
- Several chair masks contain only a backrest or seat fragment. Filtering objects at the image boundary does not catch this kind of occlusion.
- Resting or crouching poses are not automatically rejected. The lying dog `000000000074_00` remains a candidate even though its silhouette differs from a standing dog.
- Some small bottles have coarse polygon boundaries. They need a separate contour-quality check rather than being treated as clear cases of occlusion.
- Cup handles also need attention: converting polygon labels can change holes, so a recognizable outer contour does not establish that all internal geometry was preserved.

Uncertain samples should stay outside the training set until reviewed. Retained samples also need human confirmation before use in the final dataset. No contours were repaired or filled in this trial.

## Reproduce the sheets

From the repository root, first run the pilot, then:

```sh
python build_review_sheet.py --data-dir /path/to/coco128-seg
```

The script saves two source-crop comparison sheets under `output/silhouette_pilot/` and regenerates the mask-only image above. The original-photo crops stay in the local output folder. Use them with the CSV to inspect or revise the decisions.

Source: [Ultralytics COCO128-Seg](https://docs.ultralytics.com/datasets/segment/coco128-seg/), derived from [COCO](https://cocodataset.org/). This trial uses converted polygon labels; the planned full dataset will use COCO 2017 instance annotations.
