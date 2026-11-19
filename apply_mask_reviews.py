"""Exclude rejected/uncertain source masks and all their derivatives from manifests."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from prepare_dataset import SPLITS, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    singles = [json.loads(line) for line in (root/'single_samples.jsonl').read_text(encoding='utf-8').splitlines()]
    pairs = [json.loads(line) for line in (root/'overlap_samples.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id = {r['id']:r for r in singles}
    with (root/'review_queue.csv').open(encoding='utf-8',newline='') as handle:
        reviews = list(csv.DictReader(handle))
    excluded = set()
    for row in reviews:
        assert row['id'] in by_id
        assert int(row['annotation_id']) == by_id[row['id']]['annotation_id']
        assert row['decision'] in {'pending','retain','reject','uncertain'}
        if row['decision'] in {'reject','uncertain'}:
            excluded.add(row['id'])
    kept_singles = [r for r in singles if r['id'] not in excluded]
    kept_pairs = [r for r in pairs if not excluded.intersection(r['source_instances'])]
    for split in SPLITS:
        write_jsonl(root/f'{split}.jsonl',[r for r in kept_singles+kept_pairs if r['split']==split])
    write_json(root/'manifest_exclusions.json',sorted(excluded))
    summary = {'review_queue':len(reviews),'decision_counts':dict(Counter(r['decision'] for r in reviews)),
               'active_single_images':len(kept_singles),'active_overlap_images':len(kept_pairs),
               'active_total_images':len(kept_singles)+len(kept_pairs),
               'excluded_source_instances':len(excluded),'removed_overlap_images':len(pairs)-len(kept_pairs),
               'single_counts':dict(Counter(r['category'] for r in kept_singles)),
               'per_split':{s:dict(Counter(r['condition'] for r in kept_singles+kept_pairs if r['split']==s)) for s in SPLITS},
               'review_note':'Recorded decisions are preliminary visual screening. Most instances remain unreviewed.',
               'loading_rule':'Use train.jsonl, validation.jsonl and test.jsonl; do not enumerate image folders.'}
    write_json(root/'screening_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
