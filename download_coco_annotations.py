"""Download official COCO 2017 instance annotations without the 18 GB photo archive."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/coco2017'))
    args = parser.parse_args()
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'annotations_trainval2017.zip'
    target = root / 'annotations' / 'instances_train2017.json'
    if not archive.exists():
        partial = archive.with_suffix('.zip.part')
        start = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-'} if start else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            if start and response.status != 206:
                start = 0
            total = start + int(response.headers.get('Content-Length', 0))
            count, last = start, time.monotonic()
            with partial.open('ab' if start else 'wb') as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    count += len(chunk)
                    if time.monotonic() - last >= 5:
                        print(f'Downloaded {count/1024**2:.1f} / {total/1024**2:.1f} MiB', flush=True)
                        last = time.monotonic()
            if total and count != total:
                raise RuntimeError(f'Incomplete download: {count} != {total}')
        partial.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        for name in ['annotations/instances_train2017.json', 'annotations/instances_val2017.json']:
            destination = root / name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(name) as source, destination.open('wb') as sink:
                    shutil.copyfileobj(source, sink)
                print(f'Extracted {name}', flush=True)
    manifest = {'source_url': URL, 'archive_bytes': archive.stat().st_size,
                'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
                'instance_annotations': ['annotations/instances_train2017.json', 'annotations/instances_val2017.json']}
    (root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Ready: {target}', flush=True)


if __name__ == '__main__':
    main()
