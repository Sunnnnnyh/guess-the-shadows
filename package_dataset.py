"""Export only active samples, their manifests and preparation records for training."""
import argparse
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import zipfile

from prepare_dataset import SPLITS, preview, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    names = json.loads((root/'class_names.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for s in SPLITS for line in (root/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()]
    singles = [r for r in rows if r['condition']=='single']
    pairs = [r for r in rows if r['condition']!='single']
    with (root/'class_counts.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['category','train_single','validation_single','test_single','total_single'])
        for name in names:
            counts = Counter(r['split'] for r in singles if r['category']==name)
            writer.writerow([name,*(counts[s] for s in SPLITS),sum(counts.values())])
    versions = {name:importlib.metadata.version(name) for name in ['numpy','Pillow','pycocotools']}
    write_json(root/'dependency_versions.json',versions)
    preview(root,singles,pairs,names)
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    other_files = [f'{s}.jsonl' for s in SPLITS]+[
        'class_names.json','class_counts.csv','category_scan.csv','build_config.json','dataset_summary.json',
        'screening_summary.json','manifest_exclusions.json','review_queue.csv','dependency_versions.json','preview.png']
    if (root/'validation_results.json').exists():
        other_files.append('validation_results.json')
    with zipfile.ZipFile(args.archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for row in rows:
            archive.write(root/row['path'],row['path'])
        for row in singles:
            archive.write(root/row['mask_path'],row['mask_path'])
        for path in other_files:
            archive.write(root/path,path)
        for filename, selected in [('single_samples.jsonl',singles),('overlap_samples.jsonl',pairs)]:
            archive.writestr(filename,''.join(json.dumps(r)+'\n' for r in selected))
        archive.writestr('README.txt',
            'First-pass COCO silhouette candidates. Use the train/validation/test JSONL manifests, '
            'not directory enumeration. Most source masks have not yet received visual review. '
            'Rejected or uncertain reviewed sources and their derivatives are excluded. '
            'No original photographs are included. Category names remain provisional.\n')
    with zipfile.ZipFile(args.archive) as archive:
        members = set(archive.namelist())
        assert all(row['path'] in members for row in rows)
        assert not any(name.startswith('source_photos/') for name in members)
    print(f'Exported {len(singles)} single and {len(pairs)} overlap images: {args.archive}')
    print(f'Archive size: {args.archive.stat().st_size/1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
