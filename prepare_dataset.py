"""Build a source-disjoint COCO silhouette dataset and matched overlap triplets.

Automatic screening produces candidates, not human-certified complete outlines.
Original photographs and annotations are never committed to the repository.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as mask_api

DEFAULT_CATEGORIES = ['cat', 'dog', 'bird', 'bottle', 'cup', 'bus', 'car', 'airplane', 'scissors', 'teddy bear']
BANDS = {'low': (.10, .20), 'medium': (.20, .35), 'high': (.35, .50)}
SPLITS = ('train', 'validation', 'test')
CANVAS = 128


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in values:
            handle.write(json.dumps(row)+'\n')


def source_splits(image_ids, seed):
    image_ids = sorted(image_ids)
    random.Random(seed).shuffle(image_ids)
    a, b = int(len(image_ids)*.7), int(len(image_ids)*.85)
    return {key: split for split, ids in zip(SPLITS, [image_ids[:a], image_ids[a:b], image_ids[b:]]) for key in ids}


def preliminary_reason(ann, image, min_area, min_side, margin):
    if ann.get('iscrowd', 0):
        return 'crowd'
    box = ann.get('bbox', [])
    if len(box) != 4 or not np.isfinite(box).all() or min(box[2:]) <= 0:
        return 'invalid_bbox'
    x, y, w, h = box
    if min(w, h) < min_side or ann.get('area', 0) < min_area:
        return 'small'
    if x <= margin or y <= margin or x+w >= image['width']-margin or y+h >= image['height']-margin:
        return 'image_boundary'
    if not ann.get('segmentation'):
        return 'missing_segmentation'
    return None


def decode_mask(ann, image):
    seg = ann['segmentation']
    if isinstance(seg, list):
        rle = mask_api.merge(mask_api.frPyObjects(seg, image['height'], image['width']))
    elif isinstance(seg['counts'], list):
        rle = mask_api.frPyObjects(seg, image['height'], image['width'])
    else:
        rle = dict(seg)
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('ascii')
    return mask_api.decode(rle).astype(bool)


def proportional_resize(mask, scale):
    size = tuple(max(1, round(v*scale)) for v in mask.size)
    return mask.resize(size, Image.Resampling.NEAREST)


def render_single(mask):
    scaled = proportional_resize(mask, 112/max(mask.size))
    image = Image.new('L', (CANVAS, CANVAS), 255)
    image.paste(ImageOps.invert(scaled), ((CANVAS-scaled.width)//2, (CANVAS-scaled.height)//2))
    return image


def overlap_at(a, b, dx, dy):
    """Intersection with A at (0,0), B at a relative integer offset."""
    ah, aw = a.shape
    bh, bw = b.shape
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(aw, dx+bw), min(ah, dy+bh)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(a[y0:y1, x0:x1] & b[y0-dy:y1-dy, x0-dx:x1-dx]))


def matched_triplet(mask_a, mask_b, rng, attempts=1200):
    # Match foreground areas before translating; hold both sizes fixed afterwards.
    area_a, area_b = np.count_nonzero(mask_a), np.count_nonzero(mask_b)
    a = proportional_resize(mask_a, (1400/area_a)**.5)
    b = proportional_resize(mask_b, (1400/area_b)**.5)
    factor = min(1., 96/max(*a.size, *b.size), 120/min(a.width+b.width, a.height+b.height))
    a, b = proportional_resize(a, factor), proportional_resize(b, factor)
    aa, bb = np.asarray(a)>0, np.asarray(b)>0
    areas = [int(aa.sum()), int(bb.sum())]
    if min(areas) == 0 or max(areas)/min(areas) > 2:
        return None
    found = {}
    for _ in range(attempts):
        dx, dy = rng.randint(1-b.width, a.width-1), rng.randint(1-b.height, a.height-1)
        x0, y0, x1, y1 = min(0, dx), min(0, dy), max(a.width, dx+b.width), max(a.height, dy+b.height)
        if x1-x0 > CANVAS-8 or y1-y0 > CANVAS-8:
            continue
        intersection = overlap_at(aa, bb, dx, dy)
        ios = intersection/min(areas)
        band = next((name for name, (lo, hi) in BANDS.items() if lo <= ios < hi), None)
        if band is None or band in found:
            continue
        ox, oy = (CANVAS-(x1-x0))//2-x0, (CANVAS-(y1-y0))//2-y0
        positions = [(ox, oy), (ox+dx, oy+dy)]
        canvas_a, canvas_b = np.zeros((CANVAS, CANVAS), bool), np.zeros((CANVAS, CANVAS), bool)
        for canvas, array, (x, y) in zip([canvas_a, canvas_b], [aa, bb], positions):
            h, w = array.shape
            if not (0 <= x and 0 <= y and x+w <= CANVAS and y+h <= CANVAS):
                raise AssertionError('Clipping in placement')
            canvas[y:y+h, x:x+w] = array
        assert np.count_nonzero(canvas_a & canvas_b) == intersection
        found[band] = {'image': Image.fromarray(np.uint8(~(canvas_a | canvas_b))*255),
                       'ios': ios, 'intersection_pixels': intersection,
                       'mask_areas': areas, 'sizes': [list(a.size), list(b.size)],
                       'positions': [list(p) for p in positions]}
        if len(found) == 3:
            return found
    return None


def preview(out, singles, pairs, names):
    cells = []
    for category in names:
        record = next(r for r in singles if r['category'] == category)
        cells.append((record['path'], category, 'single / automatic candidate'))
    for record in pairs[:12]:
        cells.append((record['path'], ' + '.join(record['categories']), f"{record['condition']} | IoS {record['ios']:.3f}"))
    columns, width, height = 4, 280, 228
    sheet = Image.new('RGB', (columns*width, ((len(cells)+columns-1)//columns)*height+60), 'white')
    draw = ImageDraw.Draw(sheet)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    draw.text((12, 10), 'COCO 2017 | first-pass candidates and matched overlap examples', fill='black', font=font)
    for i, (path, name, note) in enumerate(cells):
        x, y = (i%columns)*width, (i//columns)*height+60
        with Image.open(out/path) as image:
            sheet.paste(image.resize((160,160), Image.Resampling.NEAREST).convert('RGB'), (x+50, y))
        draw.text((x+8, y+169), name, fill='black', font=font)
        draw.text((x+8, y+194), note, fill='#555555', font=font)
    sheet.save(out/'preview.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('output/formal_v1'))
    parser.add_argument('--categories', nargs='+', default=DEFAULT_CATEGORIES)
    parser.add_argument('--confirmed-categories', action='store_true', help='Record an already confirmed class selection')
    parser.add_argument('--instances-per-class', type=int, default=400)
    parser.add_argument('--pairs', type=int, default=3000, help='Number of distinct pairs, each at three overlap levels')
    parser.add_argument('--seed', type=int, default=360)
    parser.add_argument('--min-area', type=int, default=500)
    parser.add_argument('--min-side', type=int, default=24)
    parser.add_argument('--margin', type=int, default=2)
    parser.add_argument('--scan-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'dataset_summary.json').exists() and not args.scan_only:
        raise FileExistsError('Choose a new output folder to avoid overwriting an existing build')
    started = time.monotonic()
    print('Reading original COCO annotations...', flush=True)
    data = json.loads(args.annotations.read_text(encoding='utf-8'))
    images = {r['id']: r for r in data['images']}
    cats = {r['id']: r['name'] for r in data['categories']}
    unknown = set(args.categories)-set(cats.values())
    if unknown or len(set(args.categories)) != len(args.categories):
        raise ValueError(f'Unknown or duplicate categories: {unknown}')
    partitions = source_splits(images, args.seed)
    counts = {name: Counter() for name in cats.values()}
    pools = defaultdict(list)
    for ann in data['annotations']:
        category, image = cats[ann['category_id']], images[ann['image_id']]
        counts[category]['raw'] += 1
        reason = preliminary_reason(ann, image, args.min_area, args.min_side, args.margin)
        if reason:
            counts[category][reason] += 1
            continue
        counts[category]['passed_metadata_screen'] += 1
        split = partitions[ann['image_id']]
        counts[category]['candidate_'+split] += 1
        if category in args.categories:
            pools[category, split].append(ann)
    with (out/'category_scan.csv').open('w', encoding='utf-8', newline='') as handle:
        fields = ['category', 'raw', 'crowd', 'small', 'image_boundary', 'invalid_bbox', 'missing_segmentation',
                  'passed_metadata_screen', 'candidate_train', 'candidate_validation', 'candidate_test']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({'category': name, **{key: counts[name][key] for key in fields[1:]}} for name in cats.values())
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'source': 'COCO instances_train2017.json', 'canvas': CANVAS,
                   'overlap_bands': BANDS, 'categories_provisional': not args.confirmed_categories,
                   'screening_note': 'Metadata and raster checks cannot establish a complete amodal outline'})
    write_json(out/'build_config.json', config)
    print('Candidate counts:', {name: counts[name]['passed_metadata_screen'] for name in args.categories}, flush=True)
    if args.scan_only:
        return
    mask_dir = out/'instances'
    mask_dir.mkdir(exist_ok=True)
    singles, raster_rejections = [], Counter()
    single_quotas = [int(args.instances_per_class*.7), int(args.instances_per_class*.15)]
    single_quotas.append(args.instances_per_class-sum(single_quotas))
    for category in args.categories:
        for split, quota in zip(SPLITS, single_quotas):
            pool = pools[category, split]
            # Prefer one polygon, then use a stable hash order; this is a documented selection bias.
            pool.sort(key=lambda ann: (len(ann['segmentation']) != 1 if isinstance(ann['segmentation'], list) else False,
                                      hashlib.sha256(f"{args.seed}:{ann['id']}".encode()).hexdigest()))
            kept = 0
            for ann in pool:
                if kept == quota:
                    break
                image = images[ann['image_id']]
                try:
                    mask = decode_mask(ann, image)
                except (ValueError, TypeError, IndexError, KeyError):
                    raster_rejections['invalid_segmentation'] += 1
                    continue
                yy, xx = np.nonzero(mask)
                if len(xx) < args.min_area:
                    raster_rejections['small_raster_area'] += 1
                    continue
                x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
                if min(x1-x0, y1-y0) < args.min_side:
                    raster_rejections['small_raster_side'] += 1
                    continue
                if x0 <= args.margin or y0 <= args.margin or x1 >= image['width']-args.margin or y1 >= image['height']-args.margin:
                    raster_rejections['raster_image_boundary'] += 1
                    continue
                crop = Image.fromarray(np.uint8(mask[y0:y1, x0:x1])*255)
                instance_id = f"coco_{ann['image_id']:012d}_{ann['id']}"
                mask_path = Path('instances')/(instance_id+'.png')
                crop.save(out/mask_path)
                path = Path('images')/split/'single'/(instance_id+'.png')
                (out/path).parent.mkdir(parents=True, exist_ok=True)
                render_single(crop).save(out/path)
                row = {'id': instance_id, 'annotation_id': ann['id'], 'source_image_id': ann['image_id'],
                       'source_file_name': image['file_name'], 'source_width': image['width'], 'source_height': image['height'],
                       'source_url': f"https://s3.amazonaws.com/images.cocodataset.org/train2017/{image['file_name']}",
                       'category': category, 'label_indices': [args.categories.index(category)],
                       'split': split, 'condition': 'single', 'mask_path': mask_path.as_posix(), 'path': path.as_posix(),
                       'raster_area': len(xx), 'bbox_xyxy': [x0,y0,x1,y1], 'annotation_area': ann['area'],
                       'segmentation_parts': len(ann['segmentation']) if isinstance(ann['segmentation'], list) else None,
                       'review_status': 'unreviewed_automatic_candidate'}
                singles.append(row)
                kept += 1
            if kept != quota:
                raise RuntimeError(f'{category}/{split}: only {kept} valid masks for quota {quota}; see category_scan.csv')
        print(f'Extracted {args.instances_per_class} {category} candidate masks', flush=True)
    del data, pools
    write_jsonl(out/'single_samples.jsonl', singles)
    review_rows = []
    for category in args.categories:
        source_ids = set()
        for row in [r for r in singles if r['category'] == category and r['split']=='train']:
            if row['source_image_id'] in source_ids:
                continue
            review_rows.append({key: row[key] for key in ['id','annotation_id','source_image_id','category','split','source_url','mask_path']})
            source_ids.add(row['source_image_id'])
            if len(source_ids) == 20:
                break
    with (out/'review_queue.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0])+['decision','reason'])
        writer.writeheader()
        writer.writerows({**r,'decision':'pending','reason':''} for r in review_rows)
    by_group = defaultdict(list)
    for row in singles:
        by_group[row['category'],row['split']].append(row)
    masks = {r['id']: Image.open(out/r['mask_path']).copy() for r in singles}
    combinations = [(a,b) for i,a in enumerate(args.categories) for b in args.categories[i+1:]]
    pair_quotas = [int(args.pairs*.7), int(args.pairs*.15)]
    pair_quotas.append(args.pairs-sum(pair_quotas))
    pairs, seen_pairs, failed_geometry = [], set(), 0
    rng = random.Random(args.seed)
    for split, quota in zip(SPLITS, pair_quotas):
        cycle = combinations.copy()
        rng.shuffle(cycle)
        for i in range(quota):
            ca, cb = cycle[i%len(cycle)]
            for retry in range(100):
                a, b = rng.choice(by_group[ca,split]), rng.choice(by_group[cb,split])
                key = tuple(sorted([a['id'],b['id']]))
                if key in seen_pairs or a['source_image_id'] == b['source_image_id']:
                    continue
                triplet = matched_triplet(masks[a['id']],masks[b['id']],rng)
                if triplet:
                    break
                failed_geometry += 1
            else:
                raise RuntimeError(f'Cannot build {ca}/{cb} triplet after 100 attempts')
            seen_pairs.add(key)
            pair_id = f'{split}_pair_{i:05d}'
            for condition in BANDS:
                render = triplet[condition]
                path = Path('images')/split/condition/(pair_id+'.png')
                (out/path).parent.mkdir(parents=True, exist_ok=True)
                render['image'].save(out/path)
                pairs.append({'id':pair_id+'_'+condition, 'pair_id':pair_id,
                              'source_instances':[a['id'],b['id']], 'source_image_ids':[a['source_image_id'],b['source_image_id']],
                              'categories':[ca,cb], 'label_indices':[args.categories.index(ca),args.categories.index(cb)],
                              'split':split,'condition':condition,'path':path.as_posix(),
                              **{k:v for k,v in render.items() if k!='image'},
                              'review_status':'derived_from_unreviewed_automatic_candidates'})
            if (i+1)%100 == 0:
                print(f'{split}: {i+1}/{quota} matched pairs', flush=True)
    write_jsonl(out/'overlap_samples.jsonl', pairs)
    write_json(out/'class_names.json', args.categories)
    for split in SPLITS:
        write_jsonl(out/f'{split}.jsonl', [r for r in singles+pairs if r['split']==split])
    source_sets = {split:{r['source_image_id'] for r in singles if r['split']==split} for split in SPLITS}
    assert all(not source_sets[a]&source_sets[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:])
    summary = {'status':'first-pass automatic candidates; human screening not complete',
               'categories_provisional':not args.confirmed_categories, 'categories':args.categories, 'seed':args.seed,
               'single_instances':len(singles), 'distinct_pairs':len(seen_pairs), 'overlap_images':len(pairs),
               'total_images':len(singles)+len(pairs), 'review_queue_count':len(review_rows),
               'unique_source_photographs':len(set.union(*source_sets.values())),
               'raster_rejections_during_selection':dict(raster_rejections), 'geometry_retries':failed_geometry,
               'per_split':{split:dict(Counter(r['condition'] for r in singles+pairs if r['split']==split)) for split in SPLITS},
               'source_disjoint':True, 'external_final_test_collected':False,
               'selection_note':'One-polygon masks preferred, followed by seeded hash order. Visual review remains necessary.',
               'seconds':round(time.monotonic()-started,2)}
    write_json(out/'dataset_summary.json', summary)
    preview(out, singles, pairs, args.categories)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
