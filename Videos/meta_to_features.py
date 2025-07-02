import argparse
import csv
import os
from typing import List


def convert(meta_csv: str, out_csv: str) -> None:
    rows: List[List[str]] = []
    with open(meta_csv, newline='', encoding='utf-8') as fr:
        reader = csv.DictReader(fr)
        for r in reader:
            path = r.get("relative_path") or r.get("video") or r.get("image") or ""
            norm = path.replace("\\", "/")
            name = os.path.basename(norm)
            size = int(r.get("size_bytes") or r.get("size") or 0)
            mime = r.get("mime_type") or r.get("type") or ""
            typ = mime.split("/")[0] if "/" in mime else mime
            extension = os.path.splitext(name)[1].lower()
            sensitivity = float(r.get('sensitivity') or 0.0)
            algorithm = (
                r.get("best_algorithm")
                or r.get("algorithm")
                or r.get("best_algo")
                or ""
            )
            rows.append([name, size, typ, extension, sensitivity, algorithm])
    with open(out_csv, 'w', newline='', encoding='utf-8') as fw:
        csv.writer(fw).writerows([
            ["name", "size", "type", "extension", "sensitivity", "algorithm"],
            *rows,
        ])


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Convert metadata CSV to feature CSV")
    p.add_argument('meta', help='Input metadata CSV')
    p.add_argument('--out', default='features.csv', help='Output feature CSV')
    return p.parse_args()


def main() -> None:
    args = get_args()
    convert(args.meta, args.out)


if __name__ == '__main__':
    main()
