# get_selma_cross_val_folds.py - Function to get cross-validation folds for SELMA dataset

# --- Setup ---

# imports
import argparse
import json
import math
from pathlib import Path
import random


# --- Functions ---

# function to find images
def discover_images(class_dir, channel_substr):

    # channel filter (none means to accept all)
    subs = None
    s = str(channel_substr).strip()
    if s and s.upper() != 'ALL':
        subs = [t.strip().lower() for t in s.split(',') if t.strip()]

    # list of collected image paths
    img_paths = []

    # iterate over all nifti-like files in subtype directory
    for p in sorted(class_dir.glob('*.nii*')):
        low = p.name.lower()

        # skip label files
        if low.endswith('_label.nii') or low.endswith('_label.nii.gz'):
            continue

        # check channel substrings
        if subs is not None and not any(tok in low for tok in subs):
            continue

        # store absolute path
        img_paths.append(str(p.resolve()))

    return img_paths


# function to get cross-validation folds for SELMA dataset
def get_folds(paths, train_limit, repeats, seed):

    # create deterministic permutation of paths
    rng = random.Random(seed)
    paths_ordered = sorted(paths)
    rng.shuffle(paths_ordered)

    # sanity checks
    n = len(paths_ordered)
    if n == 0:
        raise ValueError('No image paths found')
    if train_limit <= 0 or train_limit > n:
        raise ValueError(f'Invalid train_limit: {train_limit}, must be in [1, {n}]')
    
    # ensure at least 1 test item exists
    test_size = n - train_limit
    if test_size < 1:
        raise ValueError(f'Not enough samples ({n}) for the specified train_limit ({train_limit}), at least 1 test sample is required')
    
    # list for folds
    folds = []

    # create folds by rotating window of length test_size
    for r in range(repeats):

        # sliding start index
        start = (r * test_size) % n

        # collect test block indices [start, start + test_size]
        test_block = [paths_ordered[(start + i) % n] for i in range(test_size)]
        test_set = set(test_block)

        # collect train block indices (all not in test block)
        train_set = [p for p in paths_ordered if p not in test_set]

        # validate sizes
        assert len(train_set) == train_limit and len(test_block) == test_size

        # store fold
        folds.append({'train': train_set, 'eval': test_block})

    return folds, test_size


# main function
def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Root with subtype folders')
    parser.add_argument('--subtypes', nargs='*', default=['ALL'], help='List of subtypes (subdirs) to process (default: all subdirs)')
    parser.add_argument('--exclude_subtypes', nargs='*', default=[], help='List of subtypes (subdirs) to exclude (default: none)')
    parser.add_argument('--channel_substr', default='ALL', help='Channel filter: "ALL" or comma separated like "ch0,ch1" (default: ALL)')
    parser.add_argument('--train_limit', type=int, required=True, help='Number of training samples per fold')
    parser.add_argument('--repeats', type=int, default=10, help='Number of folds to create (default: 10)')
    parser.add_argument('--seed', type=int, default=100, help='Random seed (default: 100)')
    parser.add_argument('--output_json', required=True, help='Output JSON file to save folds')
    args = parser.parse_args()

    # root path
    root = Path(args.root)

    # resolve which subtypes to process
    if any(s.upper() == 'ALL' for s in args.subtypes):
        subtypes = sorted([d.name for d in root.iterdir() if d.is_dir() and d.name not in args.exclude_subtypes])

    # use only provided subtypes
    else:
        subtypes = args.subtypes

    # dict for all folds
    all_folds = {}

    # process each subtype
    for subtype in subtypes:

        # get available images
        imgs = discover_images(root / subtype, args.channel_substr)
        n = len(imgs)

        # ensure enough samples
        if n == 0:
            print(f'[WARN] {subtype}: No images found, skipping', flush=True)
            continue
        if args.train_limit >= n:
            print(f'[WARN] {subtype}: Not enough samples ({n}) for the specified train_limit ({args.train_limit}), skipping', flush=True)
            continue

        # build repeated folds and get the test size (N - K)
        try:
            folds, test_size = get_folds(imgs, args.train_limit, args.repeats, args.seed)
        # if constraints unsatisfied, skip
        except Exception as e:
            print(f'[WARN] {subtype}: Error creating folds: {e}, skipping', flush=True)
            continue

        # coverage report
        min_repeats_full = math.ceil(n / test_size) # minimum repeats to ensure each sample is in test at least once
        coverage = 'FULL' if args.repeats >= min_repeats_full else 'PARTIAL'
        print(f'[INFO] {subtype}: N={n} | train_limit={args.train_limit} | test_size={test_size} | repeats={args.repeats} | coverage={coverage} (min_repeats_full={min_repeats_full})', flush=True)

        # store folds
        all_folds[subtype] = {
            'folds': folds, # list of train+eval dicts
            'repeats': args.repeats, # number of repeats requested
            'n_images': n, # total images available
            'train_limit': args.train_limit, # K (size of train pool)
            'test_size': test_size, # N - K
            'coverage': coverage, # FULL or PARTIAL
            'min_repeats_full': min_repeats_full # minimum repeats for full coverage
        }

    # ensure parent dir exists for output
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save to output json
    with open(out_path, 'w') as f:
        json.dump(all_folds, f, indent=4)

    print(f'[INFO] Saved folds for {len(all_folds)} subtypes to {args.output_json}', flush=True)


# --- Main Entry Point ---
if __name__ == '__main__':
    main()














