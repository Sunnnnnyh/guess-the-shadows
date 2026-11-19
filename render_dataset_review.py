"""Download only review-queue source photos and compare their crops with masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont, ImageOps


def fetch(row, folder):
    destination = folder/(str(row['source_image_id']).zfill(12)+'.jpg')
    if destination.exists():
        with Image.open(destination) as image:
            image.verify()
        return
    url = row['source_url']
    if not url.startswith('https://s3.amazonaws.com/images.cocodataset.org/train2017/'):
        raise ValueError('Unexpected source-photo URL')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                payload = response.read()
            temporary = destination.with_suffix('.jpg.part')
            temporary.write_bytes(payload)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(destination)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    root = args.dataset.resolve()
    records = {r['id']:r for r in map(json.loads, (root/'single_samples.jsonl').read_text(encoding='utf-8').splitlines())}
    with (root/'review_queue.csv').open(encoding='utf-8', newline='') as handle:
        selected = [{**records[r['id']], 'decision':r['decision']} for r in csv.DictReader(handle)]
    photo_dir = root/'source_photos'
    photo_dir.mkdir(exist_ok=True)
    if args.download:
        unique = {r['source_image_id']:r for r in selected}
        done = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(fetch, row, photo_dir) for row in unique.values()]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done%20 == 0:
                    print(f'Source photos: {done}/{len(unique)}', flush=True)
    review_dir = root/'review_sheets'
    review_dir.mkdir(exist_ok=True)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    index = []
    for page in range((len(selected)+9)//10):
        sheet = Image.new('RGB', (1200, 1420), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), 'COCO 2017 | source crop and extracted mask | decisions in review_queue.csv', fill='black', font=font)
        for j, row in enumerate(selected[page*10:page*10+10]):
            number = page*10+j+1
            x, y = (j%2)*600, (j//2)*270+55
            with Image.open(photo_dir/(str(row['source_image_id']).zfill(12)+'.jpg')) as source:
                x0,y0,x1,y1 = row['bbox_xyxy']
                pad = max(20,round(max(x1-x0,y1-y0)*.2))
                box = (max(0,x0-pad),max(0,y0-pad),min(source.width,x1+pad),min(source.height,y1+pad))
                crop = source.crop(box).convert('RGB')
            ImageDraw.Draw(crop).rectangle((x0-box[0],y0-box[1],x1-box[0]-1,y1-box[1]-1),outline='red',width=2)
            draw.text((x+8,y),f"{number}. {row['category']} | annotation {row['annotation_id']} | {row['decision']}",fill='black',font=font)
            draw.text((x+8,y+23),f"image {row['source_image_id']} | source area {row['raster_area']} px",fill='#555555',font=small)
            sheet.paste(ImageOps.contain(crop,(310,210)),(x+8,y+47))
            with Image.open(root/row['mask_path']) as mask:
                silhouette = ImageOps.invert(mask.convert('L'))
                sheet.paste(ImageOps.contain(silhouette,(240,210),Image.Resampling.NEAREST),(x+340,y+47))
            index.append({'number':number,'sheet':page+1,'id':row['id'],'annotation_id':row['annotation_id'],
                          'source_image_id':row['source_image_id'],'category':row['category']})
        sheet.save(review_dir/f'source_masks_{page+1:02d}.jpg',quality=90)
    (root/'review_index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    print(f'Ready: {len(selected)} mask comparisons in {page+1} sheets at {review_dir}',flush=True)


if __name__ == '__main__':
    main()
