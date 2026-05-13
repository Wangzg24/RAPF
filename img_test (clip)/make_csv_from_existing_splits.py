import argparse
import json
import re
from pathlib import Path

import pandas as pd

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def clean_text(x: str) -> str:
    x = x.replace('_', ' ').replace('-', ' ')
    x = re.sub(r'\s+', ' ', x).strip()
    return x


def is_synset_id(x: str) -> bool:
    return re.fullmatch(r'[nvars]\d{8}', x) is not None


def synset_to_text(x: str) -> str:
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.synsets('dog')
    except LookupError:
        print('[Info] Downloading WordNet resources ...')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    pos = x[0]
    offset = int(x[1:])
    syn = wn.synset_from_pos_and_offset(pos, offset)
    return clean_text(syn.lemma_names()[0])


def infer_label_text(class_name: str, use_wordnet: bool) -> str:
    if use_wordnet and is_synset_id(class_name):
        try:
            mapped = synset_to_text(class_name)
            print(f'[OK] {class_name} -> {mapped}')
            return mapped
        except Exception as e:
            print(f'[Warning] Failed to map {class_name}: {e}. Fallback to raw class name.')
            return class_name
    return clean_text(class_name)


from typing import Optional, Dict

def build_one_csv(
    split_dir: Path,
    split_name: str,
    out_csv: Path,
    use_wordnet: bool,
    label_map: Optional[Dict[str, str]] = None
):
    if not split_dir.exists():
        raise FileNotFoundError(f'Split dir not found: {split_dir}')

    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f'No class subfolders found under: {split_dir}')

    rows = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        if label_map is not None and class_name in label_map:
            label_text = label_map[class_name]
        else:
            label_text = infer_label_text(class_name, use_wordnet=use_wordnet)

        image_files = sorted(
            [p for p in class_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]
        )

        for img_path in image_files:
            rel_path = img_path.relative_to(split_dir)
            filename = str(Path(split_name) / rel_path).replace('\\', '/')
            rows.append({
                'filename': filename,
                'label': class_name,
                'label_text': label_text,
            })

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f'Saved: {out_csv}')
    print(f'  images={len(df)}, classes={df["label"].nunique()}')
    print(df.head())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录，要求其下有 images/train, images/val, images/test')
    parser.add_argument('--use_wordnet', action='store_true',
                        help='如果类别文件夹名是 synset id（如 n02110341），尝试自动映射成可读标签词')
    parser.add_argument('--label_map_json', type=str, default='',
                        help='可选，自定义映射 json，优先级高于自动映射')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    images_root = data_root / 'images'
    split_root = data_root / 'split'

    label_map = None
    if args.label_map_json:
        with open(args.label_map_json, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        print(f'[Info] Loaded label map from: {args.label_map_json}')

    for split_name in ['train', 'val', 'test']:
        split_dir = images_root / split_name
        out_csv = split_root / f'{split_name}.csv'
        build_one_csv(split_dir, split_name, out_csv, use_wordnet=args.use_wordnet, label_map=label_map)

    print('\nDone. You can now use the original training code directly.')


if __name__ == '__main__':
    main()
