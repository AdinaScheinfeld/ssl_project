# get_classification_cross_val_folds.py - Python script to get stratified cross-validation folds for classification datasets

# --- Setup ---

# imports
import argparse
import glob
import json
from pathlib import Path
import random

# --- Functions ---

# function to discover images and return dict of class to image paths
def discover_images_by_class(root_dir, 
                             include_subtypes=None, exclude_subtypes=None, 
                             channel_substr='ALL',
                             extra_class_globs=None):

    # get class directories
    class_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    all_names = [d.name for d in class_dirs]

    # filtering
    if include_subtypes is None or any(s.upper() == 'ALL' for s in include_subtypes):
        names = all_names
    else:
        names = [n for n in all_names if n in set(include_subtypes)]

    # exclude subtypes
    if exclude_subtypes is not None:
        names = [n for n in names if n not in set(exclude_subtypes)]

    # normalize channel filters
    subs = None
    s = str(channel_substr).strip()
    if s and s.upper() != 'ALL':
        subs = [t.strip().lower() for t in s.split(',') if t.strip()]

    # dict to hold class to image paths
    class_to_paths = {n: [] for n in names}

    # iterate over class directories
    for cname in class_to_paths.keys():
        dd = root_dir / cname
        for p in sorted(dd.glob('*.nii*')):
            low = p.name.lower()

            # skip label files
            if low.endswith('_label.nii') or low.endswith('_label.nii.gz'):
                continue

            # check channel substrings
            if subs is not None and not any(tok in low for tok in subs):
                continue

            # add to dict
            class_to_paths[cname].append(str(p.resolve()))

    # handle extra class globs
    extra_class_globs = extra_class_globs or []
    for spec in extra_class_globs:
        if ':' not in spec:
            print(f'[WARN] Invalid extra_class_glob spec "{spec}", skipping.', flush=True)
            continue

        # get class name and glob pattern
        cname, pattern = spec.split(':', 1)
        cname = cname.strip()
        pattern = pattern.strip()
        if not cname or not pattern:
            print(f'[WARN] Invalid extra_class_glob spec "{spec}", skipping.', flush=True)
            continue

        # ensure key exists
        if cname not in class_to_paths:
            class_to_paths[cname] = []

        # glob for files
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f'[WARN] No matches found for extra_class_glob pattern "{pattern}" for class "{cname}".', flush=True)
            continue

        for p in matches:
            p_path = Path(p)
            low = p_path.name.lower()

            # channel filtering
            if subs is not None and not any(tok in low for tok in subs):
                continue
        
            # add to dict
            resolved_path = str(p_path.resolve())
            if resolved_path not in class_to_paths[cname]:
                class_to_paths[cname].append(resolved_path)

    return class_to_paths
    
# function to get stratified cross-validation folds (each fold has balanced class distribution and is a dict with 'train' and 'test' keys)
def stratified_cv_folds(paths_by_class, repeats, seed=100, test_frac=0.2, lock_test=False, 
                        fixed_test_per_class=None, train_per_class=None):

    # set random seed
    rng = random.Random(seed)

    # sort class names
    class_names = sorted(paths_by_class.keys())

    # suffle paths within each class
    per_class_paths = {}
    for cname in class_names:
        arr_sorted = sorted(paths_by_class[cname])
        rng.shuffle(arr_sorted)
        per_class_paths[cname] = arr_sorted

    # function to split paths into folds
    def split_one(rep):

        # create lists for traina nd test file paths
        train, test = [], []

        # iterate over classes
        for cname in class_names:

            # get paths for this class
            arr = per_class_paths[cname]

            # determine number of test samples
            n_total = len(arr)

            # if no samples, skip
            if n_total == 0:
                continue

            # determine test count
            if fixed_test_per_class is not None:
                k = max(1, min(int(fixed_test_per_class), n_total - 1)) # at least 1, at most total - 1
            else:
                k = max(1, int(round(n_total * test_frac))) # at least 1

            # if lock test, use fixed split
            if lock_test:
                test_block = arr[:k]

            # otherwise, random sample
            else:
                start = (rep * k) % n_total
                test_block = [arr[(start + i) % n_total] for i in range(k)]
            test_set = set(test_block)
            
            # get train block
            pool = [p for p in arr if p not in test_set]

            # optionally limit train samples per class
            if train_per_class is not None:
                ktr = min(int(train_per_class), len(pool)) # at most available samples
                train_block = pool[:ktr]
            else:
                train_block = pool

            # add to train and test lists
            train += train_block
            test += list(test_block)

        # shuffle
        rng.shuffle(train)
        rng.shuffle(test)

        # return train and test splits
        return {'train': train, 'test': test}

    # create folds
    folds = [split_one(rep) for rep in range(repeats)]

    return folds


# --- Main ---

# main function
def main():

    # parse command line args
    parser = argparse.ArgumentParser(description='Get stratified cross-validation folds for classification datasets.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing class subdirectories.')
    parser.add_argument('--subtypes', nargs='*', default=['ALL'], help='List of class subtypes to include (default: ALL).')
    parser.add_argument('--exclude_subtypes', nargs='*', default=[], help='List of class subtypes to exclude (default: none).')
    parser.add_argument('--channel_substr', type=str, default='ALL', help='Comma-separated substrings to filter image channels (default: ALL).')
    parser.add_argument('--repeats', type=int, default=5, help='Number of cross-validation repeats (default: 5).')
    parser.add_argument('--test_frac', type=float, default=0.2, help='Fraction of data to use for testing (default: 0.2).')
    parser.add_argument('--fixed_test_per_class', type=int, default=None, help='Fixed number of test samples per class (overrides test_frac if set; caps at available).')
    parser.add_argument('--train_per_class', type=int, default=None, help='Fixed number of train samples per class (caps at available; default: all).')
    parser.add_argument('--lock_test', action='store_true', help='Lock test sets across repeats (default: False).')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility (default: 100).')
    parser.add_argument('--extra_class_globs', type=str, action='append', default=[], help='Extra class glob specifications in the format ClassName:glob_pattern (ex: VIP_ASLM_off:"/midtier/.../all_mesospim_patches/*VIP_ASLM_off*.nii*").')
    parser.add_argument('--output_json', type=str, required=True, help='Output JSON file to save the folds.')
    args = parser.parse_args()

    # convert root dir to Path
    root_dir = Path(args.root_dir)

    # discover images by class
    paths_by_class = discover_images_by_class(
        root_dir,
        include_subtypes=args.subtypes,
        exclude_subtypes=args.exclude_subtypes,
        channel_substr=args.channel_substr,
        extra_class_globs=args.extra_class_globs
    )

    # get stratified cross-validation folds
    folds = stratified_cv_folds(
        paths_by_class,
        repeats=args.repeats,
        seed=args.seed,
        test_frac=args.test_frac,
        lock_test=args.lock_test,
        fixed_test_per_class=args.fixed_test_per_class,
        train_per_class=args.train_per_class
    )

    # metadata
    metadata = {
        'classes': sorted(paths_by_class.keys()),
        'repeats': args.repeats,
        'test_frac': args.test_frac,
        'fixed_test_per_class': args.fixed_test_per_class,
        'train_per_class': args.train_per_class,
        'lock_test': args.lock_test,
        'seed': args.seed,
        'counts': {k: len(v) for k, v in paths_by_class.items()},
        'folds': folds
    }

    # save to JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # print message
    print(f'[INFO] Saved global cross-validation folds to {output_path} for {len(metadata["classes"])} classes.')


# run main
if __name__ == '__main__':
    main()










