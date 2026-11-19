"""Reuse source-annotation review decisions when rebuilding selected categories."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,nargs='+',required=True)
    parser.add_argument('--photo-cache',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    known={}
    for path in args.reviews:
        with path.open(encoding='utf-8',newline='') as handle:
            for row in csv.DictReader(handle):
                if row['decision']=='pending':
                    continue
                if row['id'] in known:
                    assert known[row['id']]['decision']==row['decision'], 'Conflicting review decisions'
                known[row['id']]=row
    queue=args.dataset/'review_queue.csv'
    with queue.open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle)
        fields=list(reader.fieldnames)
        if 'initial_reviewer' not in fields:
            fields.append('initial_reviewer')
        rows=list(reader)
    transferred=0
    photo_dir=args.dataset/'source_photos'
    photo_dir.mkdir(exist_ok=True)
    for row in rows:
        previous=known.get(row['id'])
        row['initial_reviewer']=''
        if previous:
            assert previous['annotation_id']==row['annotation_id'] and previous['category']==row['category']
            for key in ['decision','reason','initial_reviewer']:
                row[key]=previous.get(key,'')
            transferred+=1
        name=str(row['source_image_id']).zfill(12)+'.jpg'
        if not (photo_dir/name).exists():
            for folder in args.photo_cache:
                if (folder/name).exists():
                    shutil.copyfile(folder/name,photo_dir/name)
                    break
    with queue.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary={'transferred_reviews':transferred,'queue_size':len(rows),
             'matching_rule':'Exact instance ID, original annotation ID and category',
             'source_review_files':[str(p) for p in args.reviews]}
    (args.dataset/'review_transfer.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(f'Transferred {transferred}/{len(rows)} decisions without re-reviewing unchanged annotations')


if __name__=='__main__':
    main()
