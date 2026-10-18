"""Inspect COCO128-Seg and generate reproducible silhouette previews, not training results."""
from pathlib import Path
from collections import Counter
import csv
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "pilot" / "coco128-seg"
OUT = ROOT / "output" / "silhouette_pilot"
SIZE = 128
SEED = 360
NAMES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,"
    "scissors,teddy bear,hair drier,toothbrush"
).split(",")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "instances").mkdir(exist_ok=True)
    rng = random.Random(SEED)
    paths = sorted((DATA / "images" / "train2017").glob("*.jpg"))
    if not paths:
        raise FileNotFoundError("Download and extract coco128-seg.zip under data/pilot first")
    ids = [p.stem for p in paths]
    rng.shuffle(ids)
    first, second = int(len(ids) * .7), int(len(ids) * .85)
    partitions = {key: split for split, subset in (
        ("train", ids[:first]), ("validation", ids[first:second]), ("test", ids[second:])
    ) for key in subset}
    raw, usable = Counter(), Counter()
    records = []
    masks = {}
    for path in paths:
        label = DATA / "labels" / "train2017" / (path.stem + ".txt")
        if not label.exists():
            continue
        with Image.open(path) as original:
            w, h = original.size
        for index, line in enumerate(label.read_text().splitlines()):
            values = line.split()
            if not values:
                continue
            cls = int(values[0])
            raw[NAMES[cls]] += 1
            coords = np.asarray(values[1:], dtype=float)
            if len(coords) < 6 or len(coords) % 2 or not np.isfinite(coords).all():
                continue
            if (coords < 0).any() or (coords > 1).any():
                continue
            points = coords.reshape(-1, 2) * (w - 1, h - 1)
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).polygon([tuple(p) for p in points], fill=255)
            box = mask.getbbox()
            if box is None:
                continue
            x0, y0, x1, y1 = box
            area = int(np.count_nonzero(np.asarray(mask)))
            # Conservative automatic screen; internal occlusion still needs visual inspection.
            if min(x1 - x0, y1 - y0) < 24 or area < 500:
                continue
            if x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1:
                continue
            cropped = mask.crop(box)
            usable[NAMES[cls]] += 1
            key = f"{path.stem}_{index:02d}"
            cropped.save(OUT / "instances" / (key + ".png"))
            masks[key] = cropped
            records.append({"id": key, "source_image": path.stem,
                            "source_polygon_index": index, "class_id": cls,
                            "category": NAMES[cls], "split": partitions[path.stem],
                            "area": area, "bbox": list(box)})

    with (OUT / "class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "raw_instances", "passed_automatic_screen"])
        for category in NAMES:
            writer.writerow([category, raw[category], usable[category]])

    def resize(mask, longest):
        ratio = longest / max(mask.size)
        return mask.resize(tuple(max(1, round(s * ratio)) for s in mask.size), Image.Resampling.NEAREST)

    def place(mask, x, y):
        canvas = Image.new("L", (SIZE, SIZE), 0)
        canvas.paste(mask, (x, y))
        return np.asarray(canvas) > 0

    # Manually inspected illustrations. These do NOT estimate class feasibility or accuracy.
    # Largest area alone retained badly occluded chairs/bicycles; visual review is necessary.
    preview_ids = ["000000000575_00", "000000000490_00", "000000000283_00",
                   "000000000605_01", "000000000308_12", "000000000491_00",
                   "000000000143_06", "000000000081_00"]
    record_by_id = {r["id"]: r for r in records}
    selected = [record_by_id[key] for key in preview_ids if key in record_by_id]

    panels, compositions = [], []
    for record in selected:
        mask = resize(masks[record["id"]], 96)
        foreground = place(mask, (SIZE - mask.width) // 2, (SIZE - mask.height) // 2)
        panels.append((Image.fromarray(np.uint8(~foreground) * 255), record["category"], "single object"))

    pairs = [("cat", "bottle"), ("cat", "umbrella"),
             ("dog", "teddy bear"), ("bird", "cup")]
    by_category = {r["category"]: r for r in selected}
    for a_name, b_name in pairs:
        if a_name not in by_category or b_name not in by_category:
            continue
        a_rec, b_rec = by_category[a_name], by_category[b_name]
        if a_rec["split"] != b_rec["split"]:
            raise ValueError("Cannot combine instances across source partitions")
        accepted = None
        for _ in range(2000):
            a = resize(masks[a_rec["id"]], rng.randint(86, 104))
            b = resize(masks[b_rec["id"]], rng.randint(86, 104))
            ax, ay = rng.randint(0, SIZE - a.width), rng.randint(0, SIZE - a.height)
            bx, by = rng.randint(0, SIZE - b.width), rng.randint(0, SIZE - b.height)
            ma, mb = place(a, ax, ay), place(b, bx, by)
            aa, ab = int(ma.sum()), int(mb.sum())
            intersection = int((ma & mb).sum())
            ra, rb = intersection / aa, intersection / ab
            if .12 <= ra <= .45 and .12 <= rb <= .45 and .4 <= aa / ab <= 2.5:
                accepted = (ma, mb, ra, rb, a, b, ax, ay, bx, by)
                break
        if accepted is None:
            continue
        ma, mb, ra, rb, a, b, ax, ay, bx, by = accepted
        composite = Image.fromarray(np.uint8(~(ma | mb)) * 255)
        filename = f"overlap_{a_name.replace(' ', '_')}_{b_name.replace(' ', '_')}.png"
        composite.save(OUT / filename)
        compositions.append({"file": filename, "source_instances": [a_rec["id"], b_rec["id"]],
                             "categories": [a_name, b_name], "split": a_rec["split"], "seed": SEED,
                             "transforms": [{"size": list(a.size), "xy": [ax, ay]},
                                            {"size": list(b.size), "xy": [bx, by]}],
                             "overlap_fraction_per_instance": [ra, rb],
                             "exposed_fraction_per_instance": [1-ra, 1-rb]})
        panels.append((composite, f"{a_name} + {b_name}", f"overlap: {ra:.0%} / {rb:.0%}"))

    cell_w, cell_h, cols = 250, 218, 4
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 72), "#eef1f5")
    draw = ImageDraw.Draw(sheet)
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    title_font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    draw.text((16, 10), "Guess the Shadows | COCO128-Seg pilot", font=title_font, fill="#17243a")
    draw.text((16, 41), "Real annotation masks; synthetic composites. Preview only - no model predictions.", font=small, fill="#4a5669")
    for i, (picture, title, subtitle) in enumerate(panels):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h + 72
        draw.rounded_rectangle((x+7, y+5, x+cell_w-7, y+cell_h-6), radius=10, fill="white")
        sheet.paste(picture.resize((156, 156), Image.Resampling.NEAREST).convert("RGB"), (x+47, y+10))
        draw.text((x+15, y+170), title, font=font, fill="#17243a")
        draw.text((x+15, y+194), subtitle, font=small, fill="#4a5669")
    sheet.save(OUT / "preview.png")
    report = {"dataset": "Ultralytics COCO128-Seg", "source_url":
              "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip",
              "purpose": "data-pipeline pilot only; not a final training set or performance evaluation",
              "source_image_count": len(paths), "raw_instance_count": sum(raw.values()),
              "passed_automatic_screen_count": len(records), "seed": SEED,
              "image_partitions": dict(Counter(partitions.values())),
              "selected_preview_categories": [r["category"] for r in selected],
              "preview_selection": "Manually inspected illustrations, not a representative random sample",
              "class_counts": {name: {"raw": raw[name], "screened": usable[name]} for name in NAMES},
              "instances": records, "preview_compositions": compositions,
              "limitations": ["The pilot is too small to establish final class feasibility.",
                              "Polygon conversion can alter fine mask details and holes.",
                              "Source masks can already contain occlusion; screening is not an amodal-mask guarantee.",
                              "Exposed area does not guarantee that a human can recognize the composite."]}
    (OUT / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Source images: {len(paths)}; raw instances: {sum(raw.values())}; passed screen: {len(records)}")
    print("Preview categories:", ", ".join(report["selected_preview_categories"]))
    print("Pair previews:", len(compositions))
    print("Preview:", OUT / "preview.png")
    print("Pilot counts, not final dataset counts:")
    for r in selected:
        print(f"  {r['category']}: {raw[r['category']]} raw / {usable[r['category']]} screened")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    DATA = parser.parse_args().data_dir.resolve()
    main()
