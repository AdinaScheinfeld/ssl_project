# mask_comparisons.py - Module for comparing binary masks using various metrics

# --- Setup ---

# imports
import argparse
import csv
import nibabel as nib
import numpy as np
from pathlib import Path
import re
import tifffile as tiff


# --- Functions ---

# *** I/O helpers ***

# check if nifti
def _is_nifti(p):
    return p.suffix == '.nii' or p.name.endswith('.nii.gz')

# check if tiff
def _is_tiff(p):
    return p.suffix.lower() in ('.tif', '.tiff')

# load mask
def load_mask(path):

    if _is_nifti(path):
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)
    elif _is_tiff(path):
        arr = tiff.imread(str(path)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # squeeze to remove singleton dimensions
    arr = np.squeeze(arr)

    # ensure binary
    if arr.dtype != np.uint8:
        arr = (arr > 0.5).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)

    return arr

# save mask
def save_mask(arr, out_path):

    arr = np.squeeze(arr.astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in ('.tif', '.tiff'):
        axes = 'ZYX' if arr.ndim == 3 else 'YX'
        tiff.imwrite(str(out_path), 
                     arr, 
                     photometric='minisblack', 
                     metadata={'axes': axes}, 
                     ome=True, 
                     bigtiff=arr.nbytes > (2**32 - 1))
    else:
        nii = nib.Nifti1Image(arr, affine=np.eye(4, dtype=np.float32))
        nib.save(nii, str(out_path))


# *** Pairing logic ***

# # function to strip model suffixes
# def strip_model_suffix(name, suffix_a, suffix_b):

#     # get base
#     base = name
    
#     for suf in (suffix_a, suffix_b):

#         # handle .nii.gz
#         if base.endswith('.nii.gz'):
#             stem = base[:-7]
#             stem = re.sub(fr'_{re.escape(suf)}_pred$', '', stem)
#             return stem
#         else:
#             stem = Path(base).stem
#             stem = re.sub(fr'_{re.escape(suf)}_pred$', '', stem)
#             return stem
        
#     return Path(base).stem


# index predictions
def index_predictions(preds_dir, suffix):

    # gather paths
    pred_map = {}
    pats = list(preds_dir.glob(f'*_{suffix}_pred.nii')) \
        + list(preds_dir.glob(f'*_{suffix}_pred.nii.gz')) \
        + list(preds_dir.glob(f'*_{suffix}_pred.tif')) \
        + list(preds_dir.glob(f'*_{suffix}_pred.tiff'))
    
    # compute base name without suffix/extension
    for p in pats:
        if p.name.endswith('.nii.gz'):
            stem = p.name[:-7]
        else:
            stem = p.stem
        case = re.sub(fr'_{re.escape(suffix)}_pred$', '', stem)
        pred_map[case] = p

    return pred_map


# helper for label name adaptation
def _parse_replacements(spec):

    # parse comma-separated list of OLD->NEW rules into a list of (old, new) tuples, ex: 'image->labels_new_label,vol->mask'
    rules = []
    if not spec:
        return rules
    for item in spec.split(','):
        if '->' in item:
            old, new = item.split('->', 1)
            rules.append((old.strip(), new.strip()))
    return rules

def _apply_replacements(name, rules):
    out = name
    for old, new in rules:
        out = out.replace(old, new)
    return out


# function to find label
def find_label(labels_dir, case):

    # try common label file patterns
    candidates = [
        labels_dir / f'{case}.nii.gz',
        labels_dir / f'{case}.nii',
        labels_dir / f'{case}.tif',
        labels_dir / f'{case}.tiff',
        labels_dir / f'{case}_mask.nii.gz',
        labels_dir / f'{case}_mask.nii',
        labels_dir / f'{case}_mask.tif',
        labels_dir / f'{case}_mask.tiff',
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


# flexible find label
def find_label_flexible(labels_dir, case, replacements):

    # standard patterns
    found = find_label(labels_dir, case)
    if found:
        return found
    
    # try replacements
    alt = _apply_replacements(case, replacements)
    if alt != case:
        found = find_label(labels_dir, alt) or find_label(labels_dir, alt+'_mask')
        if found:
            return found

    # loose
    m = re.match(r'^([A-Za-z]*\d+)', case) 
    prefix = m.group(1) if m else case[:2]
    candidates = list(labels_dir.glob(f'{prefix}*'))
    if candidates:

        # prefer names that include 'label' or 'mask'
        preferred = [c for c in candidates if re.search(r'(label|mask)', c.name, re.IGNORECASE)]
        return (preferred or candidates)[0]
    
    return None


# *** Metrics ***

# safe divide
def _safe_divide(num, denom):
    return num / denom if denom > 0 else 0.0

# dice coefficient
def dice(a, b):

    a = (a > 0); b = (b > 0)
    intersection = np.count_nonzero(a & b)
    return _safe_divide(2 * intersection, np.count_nonzero(a) + np.count_nonzero(b))

# iou
def iou(a, b):
    
    a = (a > 0); b = (b > 0)
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return _safe_divide(intersection, union)

# precision
def precision(a_pred, a_true):

    tp = np.count_nonzero((a_pred > 0) & (a_true > 0))
    fp = np.count_nonzero((a_pred > 0) & (a_true == 0))
    return _safe_divide(tp, tp + fp)

# recall
def recall(a_pred, a_true):

    tp = np.count_nonzero((a_pred > 0) & (a_true > 0))
    fn = np.count_nonzero((a_pred == 0) & (a_true > 0))
    return _safe_divide(tp, tp + fn)


# *** Main ***

def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_dir', required=True, type=str, help='Directory containing predicted masks (for both files)')
    parser.add_argument('--suffix_a', required=True, type=str, help='Suffix for model A predictions (e.g., modelA)')
    parser.add_argument('--suffix_b', required=True, type=str, help='Suffix for model B predictions (e.g., modelB)')
    parser.add_argument('--labels_dir', type=str, default=None, help='Directory containing ground truth labels (optional)')
    parser.add_argument('--out_csv', type=str, default='mask_comparison_metrics.csv', help='Output CSV file for metrics')
    parser.add_argument('--save_disagreements', type=str, default=None, help='Directory to save disagreement masks (optional)')
    parser.add_argument('--disagreement_ext', choices=['nii.gz', 'nii', 'tif', 'tiff'], default='nii.gz', help='File extension for saving disagreement masks')
    parser.add_argument('--label_name_replacements', type=str, default='', help='Comma-separated list of OLD->NEW rules for adapting case names to label file names (e.g., "image->labels_new_label,vol->mask")')
    args = parser.parse_args()

    # get paths
    preds_dir = Path(args.preds_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    out_csv = Path(args.out_csv)
    label_replacements = _parse_replacements(args.label_name_replacements)

    a_map = index_predictions(preds_dir, args.suffix_a)
    b_map = index_predictions(preds_dir, args.suffix_b)

    # find common cases
    common_cases = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not common_cases:
        print(f'[ERROR] No common cases found between suffixes "{args.suffix_a}" and "{args.suffix_b}".')
        return
    
    print(f'[INFO] Found {len(common_cases)} common cases between suffixes "{args.suffix_a}" and "{args.suffix_b}".')

    # check if has labels
    has_labels = labels_dir is not None and labels_dir.exists()
    if has_labels:
        print(f'[INFO] Ground truth labels directory provided: {labels_dir}', flush=True)
    else:
        print(f'[INFO] No ground truth labels directory provided. Only comparing predictions.', flush=True)

    rows = []
    agg = {
        'dice_a': [], 'dice_b': [],
        'iou_a': [], 'iou_b': [],
        'precision_a': [], 'precision_b': [],
        'recall_a': [], 'recall_b': [],
        'a_vs_b_dice': [], 'a_vs_b_iou': []
    }

    save_diffs = args.save_disagreements is not None
    diff_dir = Path(args.save_disagreements) if save_diffs else None
    if save_diffs:
        diff_dir.mkdir(parents=True, exist_ok=True)

    # loop over cases
    for case in common_cases:

        pa = a_map[case]
        pb = b_map[case]

        # load predictions
        try:
            A = load_mask(pa)
            B = load_mask(pb)
        except Exception as e:
            print(f'[WARNING] Failed to load predictions for case {case}: {e}')
            continue

        # align shapes if needed
        if A.shape != B.shape:
            print(f'[WARNING] Shape mismatch for case {case}: {A.shape} vs {B.shape}. Attempting to align shapes.')
            minZ = min(A.shape[0], B.shape[0]) if A.ndim == 3 and B.ndim == 3 else None
            if A.ndim == B.ndim == 3 and minZ is not None:
                A = A[:minZ, :min(B.shape[1], A.shape[1]), :min(B.shape[2], A.shape[2])]
                B = B[:A.shape[0], :A.shape[1], :A.shape[2]]
            elif A.ndim == B.ndim == 2:
                A = A[:min(A.shape[0], B.shape[0]), :min(A.shape[1], B.shape[1])]
                B = B[:A.shape[0], :A.shape[1]]
            else:
                print(f'[ERROR] Cannot align shapes for case {case}: {A.shape} vs {B.shape}.')
                continue

        # label
        L = None
        if has_labels:
            lp = find_label_flexible(labels_dir, case, label_replacements)
            if lp is None:
                print(f'[WARNING] No label found for case {case}. Skipping label-based metrics.')
            else:
                try:
                    L = load_mask(lp)
                    print(f'[INFO] Loaded label for case {case} from {lp}', flush=True)
                except Exception as e:
                    print(f'[WARNING] Failed to load label for case {case}: {e}')
                    L = None

                # align label shape
                if L is not None and L.shape != A.shape:
                    print(f'[WARNING] Shape mismatch between label and predictions for case {case}: {L.shape} vs {A.shape}. Attempting to align shapes.')
                    if L.ndim == A.ndim:
                        if L.ndim == 3:
                            L = L[:A.shape[0], :A.shape[1], :A.shape[2]]
                        else:
                            L = L[:A.shape[0], :A.shape[1]]
                    else:
                        print(f'[ERROR] Cannot align label shape for case {case}: {L.shape} vs {A.shape}.')
                        L = None

        # compute metrics
        a_vs_b_dice = dice(A, B)
        a_vs_b_iou = iou(A, B)

        # 
        rec = {
            'case': case,
            'A_file': str(pa.name),
            'B_file': str(pb.name),
            'A_vs_B_dice': round(a_vs_b_dice, 5),
            'A_vs_B_iou': round(a_vs_b_iou, 5)
        }

        if L is not None:
            dice_a = dice(A, L)
            dice_b = dice(B, L)
            iou_a = iou(A, L)
            iou_b = iou(B, L)
            precision_a = precision(A, L)
            precision_b = precision(B, L)
            recall_a = recall(A, L)
            recall_b = recall(B, L)

            rec.update({
                'A_dice': round(dice_a, 5),
                'B_dice': round(dice_b, 5),
                'A_iou': round(iou_a, 5),
                'B_iou': round(iou_b, 5),
                'A_precision': round(precision_a, 5),
                'B_precision': round(precision_b, 5),
                'A_recall': round(recall_a, 5),
                'B_recall': round(recall_b, 5),
                'delta_dice_(A-B)': round(dice_a - dice_b, 5),
                'delta_iou_(A-B)': round(iou_a - iou_b, 5)
            })

            # aggregate
            agg['dice_a'].append(dice_a)
            agg['dice_b'].append(dice_b)
            agg['iou_a'].append(iou_a)
            agg['iou_b'].append(iou_b)
            agg['precision_a'].append(precision_a)
            agg['precision_b'].append(precision_b)
            agg['recall_a'].append(recall_a)
            agg['recall_b'].append(recall_b)

        agg['a_vs_b_dice'].append(a_vs_b_dice)
        agg['a_vs_b_iou'].append(a_vs_b_iou)

        rows.append(rec)

        # save disagreement masks
        if save_diffs:
            A_only = ((A > 0) & (B == 0)).astype(np.uint8)
            B_only = ((B > 0) & (A == 0)).astype(np.uint8)
            XOR = (A ^ B).astype(np.uint8)

            ext = args.disagreement_ext
            if ext in ('nii', 'nii.gz'):
                out_a = diff_dir / f'{case}_A_only.{ext}'
                out_b = diff_dir / f'{case}_B_only.{ext}'
                out_xor = diff_dir / f'{case}_A_vs_B_XOR.{ext}'
            else:
                out_a = diff_dir / f'{case}_A_only.{ext}'
                out_b = diff_dir / f'{case}_B_only.{ext}'
                out_xor = diff_dir / f'{case}_A_vs_B_XOR.{ext}'

            save_mask(A_only, out_a)
            save_mask(B_only, out_b)
            save_mask(XOR, out_xor)

    # write CSV
    if rows:
        keys = sorted(rows[0].keys(), key=lambda k: (k != 'case', k))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f'[INFO] Metrics written to {out_csv}', flush=True)

    # print summary
    def mean(lst): return float(np.mean(lst)) if lst else float('nan')
    print('\n=== Summary ===')
    if has_labels:
        print(f'Model A - Mean Dice: {mean(agg["dice_a"]):.4f}, Mean IoU: {mean(agg["iou_a"]):.4f}, Mean Precision: {mean(agg["precision_a"]):.4f}, Mean Recall: {mean(agg["recall_a"]):.4f}')
        print(f'Model B - Mean Dice: {mean(agg["dice_b"]):.4f}, Mean IoU: {mean(agg["iou_b"]):.4f}, Mean Precision: {mean(agg["precision_b"]):.4f}, Mean Recall: {mean(agg["recall_b"]):.4f}')
    print(f'Between A and B - Mean Dice: {mean(agg["a_vs_b_dice"]):.4f}, Mean IoU: {mean(agg["a_vs_b_iou"]):.4f}')
    print('================')


# main entry point
if __name__ == '__main__':
    main()















