"""Build a small, reproducible mask-review sheet from the pilot output."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--sources-only', action='store_true', help='Build source comparison sheets before recording decisions')
    args = parser.parse_args()
    out = ROOT / 'output' / 'silhouette_pilot'
    report = json.loads((out / 'pilot_report.json').read_text(encoding='utf-8'))
    selected = []
    for category in ['cat', 'dog', 'bottle', 'cup', 'chair', 'umbrella',
                     'teddy bear', 'bird', 'airplane', 'elephant', 'giraffe',
                     'bicycle', 'bear']:
        pool = sorted((r for r in report['instances'] if r['category'] == category), key=lambda r: r['id'])
        selected.extend(pool)
    (out / 'review_sample.json').write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    for page in range((len(selected) + 9) // 10):
        sheet = Image.new('RGB', (1200, 1350), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(selected[page * 10:page * 10 + 10]):
            x, y = (j % 2) * 600, (j // 2) * 270
            draw.text((x + 8, y + 6), f"{page * 10 + j + 1}. {r['category']} | {r['id']}", fill='black', font=font)
            photo = Image.open(args.data_dir / 'images' / 'train2017' / (r['source_image'] + '.jpg')).convert('RGB')
            x0, y0, x1, y1 = r['bbox']
            pad = max(15, round(max(x1-x0, y1-y0) * .15))
            cropbox = (max(0, x0-pad), max(0, y0-pad), min(photo.width, x1+pad), min(photo.height, y1+pad))
            crop = photo.crop(cropbox)
            ImageDraw.Draw(crop).rectangle((x0-cropbox[0], y0-cropbox[1], x1-cropbox[0]-1, y1-cropbox[1]-1), outline='red', width=2)
            sheet.paste(ImageOps.contain(crop, (320, 222)), (x+8, y+38))
            mask = ImageOps.invert(Image.open(out / 'instances' / (r['id'] + '.png')).convert('L'))
            mask = ImageOps.contain(mask, (235, 215), Image.Resampling.NEAREST)
            sheet.paste(mask, (x+350, y+38))
        sheet.save(out / f'review_source_sheet_{page + 1}.jpg')
    decisions_path = ROOT / 'evidence' / 'mask_review.csv'
    if decisions_path.exists() and not args.sources_only:
        with decisions_path.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
            reviews = {r['instance_id']: r for r in rows}
        assert len(rows) == len(reviews), 'Duplicate instance IDs in review CSV'
        assert set(reviews) == {r['id'] for r in selected}, 'Review IDs must match the sampled instances'
        assert all(r['decision'] in {'retain', 'reject', 'uncertain'} and r['reason'].strip() for r in rows)
        summary = {
            'dataset': report['dataset'],
            'reviewer': 'Project author',
            'initial_screening': 'AI-assisted visual inspection',
            'review_status': 'Source-photo comparisons checked and recorded decisions confirmed by project author',
            'selection': 'All automatically screened candidates in the 13 listed categories; not a random COCO sample',
            'reviewed_instances': len(selected),
            'unique_source_photographs': len({r['source_image'] for r in selected}),
            'decision_counts': dict(Counter(r['decision'] for r in rows)),
            'by_category': {category: dict(Counter(reviews[r['id']]['decision'] for r in selected if r['category'] == category))
                            for category in dict.fromkeys(r['category'] for r in selected)},
        }
        (ROOT / 'evidence' / 'review_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        sheet = Image.new('RGB', (1200, ((len(selected)+3)//4)*270+80), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 12), f'COCO128-Seg: {len(selected)} masks checked', fill='black', font=font)
        draw.text((18, 38), 'Source-photo comparisons checked and decisions confirmed by the project author.', fill='black', font=font)
        colors = {'retain': '#176943', 'reject': '#a52626', 'uncertain': '#886000'}
        for i, r in enumerate(selected):
            x, y = (i % 4) * 300, (i // 4) * 270 + 75
            review = reviews[r['id']]
            draw.text((x+12, y+4), f"{i+1}. {r['category']}", fill='black', font=font)
            mask = ImageOps.invert(Image.open(out / 'instances' / (r['id'] + '.png')).convert('L'))
            mask = ImageOps.contain(mask, (245, 175), Image.Resampling.NEAREST)
            sheet.paste(mask, (x + (300-mask.width)//2, y+36))
            draw.text((x+12, y+217), review['decision'].upper(), fill=colors[review['decision']], font=font)
            draw.text((x+12, y+241), r['id'], fill='#555555', font=font)
        sheet.save(ROOT / 'evidence' / 'mask_review.png')
        for page in range((len(selected)+19)//20):
            start_row = page*5
            row_count = min(5, (len(selected)+3)//4-start_row)
            detail = Image.new('RGB', (1200, row_count*270+80), 'white')
            detail.paste(sheet.crop((0, 0, 1200, 75)), (0, 0))
            detail.paste(sheet.crop((0, 75+start_row*270, 1200, 75+(start_row+row_count)*270)), (0, 75))
            detail.save(ROOT / 'evidence' / f'mask_review_page_{page+1}.png')
    print('\n'.join(f"{i+1}: {r['id']} {r['category']}" for i, r in enumerate(selected)))


if __name__ == '__main__':
    main()
