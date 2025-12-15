#!/usr/bin/env python3

# /home/ads4015/ssl_project/zeroshot_inference/segmentation/build_zeroshot_tasks.py - Build zero-shot segmentation tasks list from dataset directory.

from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    imgs = []
    for p in root.rglob('*.nii*'):
        if '_label' in p.name:
            continue
        imgs.append(p)

    with open(out, 'w') as f:
        for p in imgs:
            f.write(str(p) + '\n')

    print(f"[INFO] Wrote {len(imgs)} tasks to {out}", flush=True)

if __name__ == '__main__':
    main()
