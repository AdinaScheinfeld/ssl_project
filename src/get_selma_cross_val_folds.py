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
def get_folds(paths, train_limit, repeats, seed, test_size=None, lock_test=False):

    # create deterministic permutation of paths
    rng = random.Random(seed)
    paths_ordered = sorted(paths)
    rng.shuffle(paths_ordered)

    # sanity checks
    n = len(paths_ordered)
    if n == 0:
        raise ValueError('No image paths found')
    
    # interpret train_limit <0 as no limit
    if train_limit is None:
        train_limit = -1

    # determine test_size / effective train_limit
    if train_limit < 0:

        # compute test size
        if test_size is None:
            test_size = max(1, int(round(0.2 * n))) # at least 1 sample
        if not (1 <= test_size < n):
            raise ValueError(f'Invalid test_size: {test_size}, must be in [1, {n-1}]')
        effective_train_limit = n - test_size
    
    # if train_limit is specified, derive test_size if needed
    else:
        if train_limit > n:
            raise ValueError(f'Invalid train_limit: {train_limit}, must be at most {n}')
        if test_size is None:
            test_size = n - train_limit
        effective_train_limit = train_limit


    # ensure test size is valid
    if test_size < 1:
        raise ValueError(f'Not enough samples ({n}) for the specified train_limit ({train_limit}), at least 1 test sample is required')
    if effective_train_limit > n - test_size:
        raise ValueError(f'Invalid effective_train_limit: {effective_train_limit}, must be at most {n - test_size} to allow for test_size={test_size}')
    
    # list for folds
    folds = []

    # create folds by rotating window of length test_size
    for r in range(repeats):

        # if lock_test, use the same test set for all folds
        if lock_test:
            test_block = paths_ordered[:test_size]
        
        # otherwise, use sliding window
        else:
            # sliding start index
            start = (r * test_size) % n

            # collect test block indices [start, start + test_size]
            test_block = [paths_ordered[(start + i) % n] for i in range(test_size)]
        
        # convert to set for fast lookup
        test_set = set(test_block)

        # collect train block indices (all not in test block)
        train_pool = [p for p in paths_ordered if p not in test_set]
        train_set = train_pool[:effective_train_limit]

        # validate sizes
        if len(test_block) != test_size or len(train_set) != effective_train_limit:
            raise ValueError(f'Fold {r}: Invalid fold sizes, expected test_size={test_size} and train_limit={effective_train_limit}, got test_size={len(test_block)} and train_size={len(train_set)}')

        # store fold
        folds.append({'train': train_set, 'eval': test_block})

    return folds, test_size, effective_train_limit


# main function
def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Root with subtype folders')
    parser.add_argument('--subtypes', nargs='*', default=['ALL'], help='List of subtypes (subdirs) to process (default: all subdirs)')
    parser.add_argument('--exclude_subtypes', nargs='*', default=[], help='List of subtypes (subdirs) to exclude (default: none)')
    parser.add_argument('--channel_substr', default='ALL', help='Channel filter: "ALL" or comma separated like "ch0,ch1" (default: ALL)')
    parser.add_argument('--train_limit', type=int, default=-1, help='Number of training samples per fold (default: -1, i.e., all available)')
    parser.add_argument('--repeats', type=int, default=10, help='Number of folds to create (default: 10)')
    parser.add_argument('--test_size', type=int, default=None, help='Fixed number of test samples per fold (default: N - train_limit)')
    parser.add_argument('--lock_test', action='store_true', help='If set, use the same test set for all folds (default: False)')
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
        # when train_limit is specified, ensure enough samples
        if args.train_limit is not None and args.train_limit > 0 and args.test_size is not None:
            if args.train_limit > n - args.test_size:
                print(f'[WARN] {subtype}: Not enough samples ({n}) for the specified train_limit ({args.train_limit}) and test_size ({args.test_size}), skipping', flush=True)
                continue

        # build repeated folds and get the test size (N - K)
        try:
            folds, test_size, effective_train_limit = get_folds(
                imgs,
                train_limit=args.train_limit, 
                repeats=args.repeats, 
                seed=args.seed,
                test_size=args.test_size,
                lock_test=args.lock_test
            )

        # if constraints unsatisfied, skip
        except Exception as e:
            print(f'[WARN] {subtype}: Error creating folds: {e}, skipping', flush=True)
            continue

        # coverage report
        min_repeats_full = math.ceil(n / test_size) # minimum repeats to ensure each sample is in test at least once
        coverage = 'FULL' if args.repeats >= min_repeats_full else 'PARTIAL'
        print(f'[INFO] {subtype}: N={n} | train_limit={effective_train_limit} | test_size={test_size} | repeats={args.repeats} | coverage={coverage} (min_repeats_full={min_repeats_full})', flush=True)

        # store folds
        all_folds[subtype] = {
            'folds': folds, # list of train+eval dicts
            'repeats': args.repeats, # number of repeats requested
            'n_images': n, # total images available
            'train_limit': effective_train_limit, # K (size of train pool)
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














