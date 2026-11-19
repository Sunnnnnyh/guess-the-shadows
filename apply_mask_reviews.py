"""Exclude rejected/uncertain source masks and all their derivatives from manifests."""
import argparse
from collections import Counter, defaultdict
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
    source_sets = {s:{r['source_image_id'] for r in kept_singles if r['split']==s} for s in SPLITS}
    assert all(not source_sets[a]&source_sets[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:])
    kept_ids = {r['id'] for r in kept_singles}
    triplets = defaultdict(list)
    for row in kept_pairs:
        assert set(row['source_instances']) <= kept_ids
        assert all(by_id[k]['split']==row['split'] for k in row['source_instances'])
        assert row['categories']==[by_id[k]['category'] for k in row['source_instances']]
        triplets[row['pair_id']].append(row['condition'])
    assert all(len(v)==3 and set(v)=={'low','medium','high'} for v in triplets.values())
    write_json(root/'manifest_validation.json',{'passed':True,'source_disjoint':True,
        'excluded_sources_absent':True,'labels_match_sources':True,'complete_triplets':len(triplets),
        'active_images':len(kept_singles)+len(kept_pairs),
        'note':'Only source exclusions and manifests changed; saved image pixels were not modified.'})
    full_validation = root/'validation_results.json'
    if full_validation.exists():
        previous = json.loads(full_validation.read_text(encoding='utf-8'))
        if 'active_manifest_images' in previous:
            previous['active_manifest_images_at_last_full_pixel_check'] = previous.pop('active_manifest_images')
            write_json(full_validation, previous)
    write_json(root/'manifest_exclusions.json',sorted(excluded))
    summary = {'review_queue':len(reviews),'decision_counts':dict(Counter(r['decision'] for r in reviews)),
               'active_single_images':len(kept_singles),'active_overlap_images':len(kept_pairs),
               'active_total_images':len(kept_singles)+len(kept_pairs),
               'excluded_source_instances':len(excluded),'removed_overlap_images':len(pairs)-len(kept_pairs),
               'reviewed_instances':sum(r['decision']!='pending' for r in reviews),
               'unreviewed_single_instances':len(singles)-sum(r['decision']!='pending' for r in reviews),
               'per_category_review':{category:dict(Counter(r['decision'] for r in reviews if r['category']==category))
                                      for category in dict.fromkeys(r['category'] for r in reviews)},
               'single_counts':dict(Counter(r['category'] for r in kept_singles)),
               'per_split':{s:dict(Counter(r['condition'] for r in kept_singles+kept_pairs if r['split']==s)) for s in SPLITS},
               'review_note':'Recorded decisions are preliminary source-photo screening; instances outside this queue remain unreviewed.',
               'loading_rule':'Use train.jsonl, validation.jsonl and test.jsonl; do not enumerate image folders.'}
    write_json(root/'screening_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
