"""Check labels, image contents, matched overlap levels and source separation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_dataset import BANDS, CANVAS, SPLITS, proportional_resize, render_single, write_json


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    singles = read_jsonl(root/'single_samples.jsonl')
    overlaps = read_jsonl(root/'overlap_samples.jsonl')
    names = json.loads((root/'class_names.json').read_text(encoding='utf-8'))
    by_id = {r['id']:r for r in singles}
    assert len(by_id) == len(singles), 'Repeated single IDs'
    assert len({r['path'] for r in singles+overlaps}) == len(singles)+len(overlaps), 'Repeated image paths'
    sources = {split:{r['source_image_id'] for r in singles if r['split']==split} for split in SPLITS}
    assert all(not sources[a]&sources[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]), 'Source leakage'
    for index,row in enumerate(singles):
        assert row['label_indices'] == [names.index(row['category'])], 'Wrong single label'
        with Image.open(root/row['mask_path']) as mask:
            expected = np.asarray(render_single(mask))
        with Image.open(root/row['path']) as actual:
            np.testing.assert_array_equal(np.asarray(actual), expected)
        if (index+1)%1000 == 0:
            print(f'Checked {index+1}/{len(singles)} single images',flush=True)
    grouped = defaultdict(list)
    for row in overlaps:
        grouped[row['pair_id']].append(row)
    seen_pairs = set()
    for index,(pair_id, rows) in enumerate(grouped.items()):
        assert len(rows)==3 and {r['condition'] for r in rows}==set(BANDS), f'Incomplete triplet: {pair_id}'
        reference = rows[0]
        instances = reference['source_instances']
        key = tuple(sorted(instances))
        assert key not in seen_pairs, 'Repeated source-instance pair'
        seen_pairs.add(key)
        assert all(by_id[key]['split']==reference['split'] for key in instances), 'Pair source leakage'
        assert reference['categories']==[by_id[key]['category'] for key in instances]
        assert len(set(reference['categories']))==2
        assert reference['label_indices']==[names.index(c) for c in reference['categories']]
        resized = []
        for key, size in zip(instances,reference['sizes']):
            with Image.open(root/by_id[key]['mask_path']) as mask:
                # Reconstruct the exact two-stage nearest-neighbor resizing used in synthesis.
                intermediate = proportional_resize(mask,(1400/np.count_nonzero(mask))**.5)
                resized.append(np.asarray(intermediate.resize(tuple(size),Image.Resampling.NEAREST))>0)
        for row in rows:
            assert all(row[k]==reference[k] for k in ['source_instances','sizes','mask_areas','categories','split','label_indices'])
            assert row['source_image_ids']==[by_id[k]['source_image_id'] for k in instances]
            canvases = []
            for array, (x,y) in zip(resized,row['positions']):
                h,w = array.shape
                assert 0 <= x and 0 <= y and x+w <= CANVAS and y+h <= CANVAS, 'Clipped object'
                canvas = np.zeros((CANVAS,CANVAS),bool)
                canvas[y:y+h,x:x+w] = array
                canvases.append(canvas)
            a,b = canvases
            areas = [int(a.sum()),int(b.sum())]
            intersection = int((a&b).sum())
            ios = intersection/min(areas)
            lo,hi = BANDS[row['condition']]
            assert lo <= ios < hi, 'Incorrect overlap band'
            assert abs(ios-row['ios']) < 1e-12 and intersection==row['intersection_pixels']
            assert areas==row['mask_areas'] and max(areas)/min(areas)<=2
            with Image.open(root/row['path']) as image:
                np.testing.assert_array_equal(np.asarray(image),np.uint8(~(a|b))*255)
        if (index+1)%500 == 0:
            print(f'Checked {index+1}/{len(grouped)} overlap triplets',flush=True)
    for split in SPLITS:
        exported = read_jsonl(root/f'{split}.jsonl')
        exclusions_path = root/'manifest_exclusions.json'
        excluded = set(json.loads(exclusions_path.read_text(encoding='utf-8'))) if exclusions_path.exists() else set()
        kept_singles = [r for r in singles if r['id'] not in excluded]
        kept_overlaps = [r for r in overlaps if not excluded.intersection(r['source_instances'])]
        expected = [r for r in kept_singles+kept_overlaps if r['split']==split]
        assert exported==expected, 'Split manifest does not match sample records'
    result = {'passed':True,'single_images_checked':len(singles),'overlap_images_checked':len(overlaps),
              'matched_triplets_checked':len(grouped),'source_disjoint':True,
              'checks':['pixel contents match stored transforms','labels match source instances',
                        'fixed object sizes within each triplet','no clipped objects','area ratio at most two',
                        'IoS in the recorded band','all images binary 128x128','source photographs disjoint across splits'],
              'scope':'Geometry and metadata only; this does not certify visual mask completeness',
              'active_manifest_images':len(kept_singles)+len(kept_overlaps),
              'class_single_counts':dict(Counter(r['category'] for r in singles))}
    write_json(root/'validation_results.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
