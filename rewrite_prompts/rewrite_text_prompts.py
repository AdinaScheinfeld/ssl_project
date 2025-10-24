# rewrite_text_prompts.py - Script to use LLM to rewrite text prompts to improve pretraining data quality

# --- Setup ---

# imports
import argparse
import json
import math
import numpy as np
import os
from pathlib import Path
import random
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel, AutoTokenizer as HFEncoderTokenizer


# --- Classes and Functions ---

# wrapper that pools last_hidden_state with mean over tokens
class SimpleTextEncoder:

    # init
    def __init__(self, encoder_name='bert-base-uncased', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = HFEncoderTokenizer.from_pretrained(encoder_name)
        self.model = AutoModel.from_pretrained(encoder_name).to(self.device)
        self.model.eval()

    # embed texts
    def embed(self, texts):

        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        out = self.model(**toks)
        pooled = out.last_hidden_state.mean(dim=1)

        # l2 normalize
        embeds = torch.nn.functional.normalize(pooled, dim=1)

        return embeds
    
    # cosine similarity
    def cosine_sim(self, a, b):

        # get embeddings
        a_embeds = self.embed(a)
        b_embeds = self.embed(b)

        # cosine similarity
        cos_sim = a_embeds @ b_embeds.T

        return cos_sim.detach().cpu().numpy()
    

# --- LLM Prompt Template ---

# message to LLM for rewriting prompts
SYSTEM_MSG = (
    "You rewrite microscopy captions. Produce a single medically neutral rewrite that:\n"
    "- preserves technical meaning, modality, stains/markers, wavelengths, and class labels;\n"
    "- avoids speculation, demographics, or causal language not in the source;\n"
    "- removes subjective adjectives unless measurement-like (e.g., 'dense', 'sparse');\n"
    "- never invents new markers, regions, or claims not present in the source;\n"
    "- stays concise (1-2 sentences);\n"
    "- maybe uses a different sentence opening than the input (do NOT begin with the same 2-4 words as the input)."
)

# few shot examples (show samples of good rewrites)
FEW_SHOTS = [
    (
        "GFAP astrocytes form a beautiful network across the brain slice.",
        "GFAP immunolabeling outlines astrocytic somata and processes as filamentous structures throughout the slice."
    ),
    (
        "Light-sheet image of SYTO 24 nuclei with very bright dots everywhere.",
        "Light-sheet fluorescence image showing nuclei labeled with SYTO 24 as numerous, high-contrast puncta."
    ),
]

# function to create LLM prompt
def build_prompt(original_prompt):

    lines = [SYSTEM_MSG, "", "Examples:"]
    for src, tgt in FEW_SHOTS:
        lines.append(f'INPUT: {src}')
        lines.append(f'OUTPUT: {tgt}')
        lines.append("")
    lines.append("Now rewrite the following prompt:")
    lines.append(f"INPUT: {original_prompt.strip()}")
    lines.append("OUTPUT:")

    return "\n".join(lines)


# --- Quality Filtering Functions ---

# function to normalize whitespace and strip quotes
def postprocess_text(text):

    # regex for multiple spaces
    RE_MULTI_SPACE = re.compile(r'\s+')

    # strip whitespace
    text = text.strip()

    # remove surrounding quotes
    text = text.strip("'").strip('"')

    # replace multiple spaces with single space
    text = RE_MULTI_SPACE.sub(' ', text)

    return text


# function to filter rewritten prompts
def basic_filtering(original, rewritten, banned_terms):

    # regex for bad words
    RE_BAD = re.compile(r"\b(beautiful|lovely|gorgeous|stunning|obvious)\b", re.IGNORECASE)

    # create list to hold reasons for rejection
    reject_reasons = []

    # ensure not too short
    if len(rewritten.strip()) < 15:
        reject_reasons.append('too_short (<15 chars)')
    
    # ensure subjective adjectives not present
    if RE_BAD.search(rewritten):
        reject_reasons.append('subjective_adjectives_present')
    
    # ensure banned terms not present
    for b in banned_terms:
        if re.search(rf"\b{re.escape(b)}\b", rewritten, re.IGNORECASE):
            reject_reasons.append(f'banned_term_present: {b}')
        
    # avoid "unknown", "unannotated" if original is specific
    rw_low = rewritten.lower()
    if (('unknown' in rw_low) or ('unannotated' in rw_low)) and len(original.strip()) > 40:
        reject_reasons.append('inappropriate_unknown_for_specific_original')

    # if all checks passed, return True and return list of reasons
    return (len(reject_reasons) == 0), reject_reasons

# function to build whitelist of terms from original prompts (ex: wavelengths, antibody)
def build_whitelist_from_original(original):

    # create whitelist
    whitelist = set()

    # wavelengths (store both '### nm' and '###' forms)
    for m in re.findall(r"\b(4\d{2}|5\d{2}|6\d{2}|7\d{2})\s*nm\b", original):
        whitelist.add(f'{m} nm')
        whitelist.add(f'{m}')

    # antibodies/markers (simple heuristic of uppercase words >= 3 letters, ex: CTIP2, GFAP, LYVE1, DBH, etc)
    for m in re.findall(r"\b([A-Z0-9]{3,})\b", original):
        whitelist.add(m)

    return sorted(whitelist)

# function to reject rewrites that introduce new all caps marker-like tokens that were'nt in the original
def introduces_new_markers(original_whitelist, rewritten):

    # find tokens that look like markers/wavelengths (3+ A-Z or digits)
    new_tokens = set(re.findall(r"\b[A-Z0-9]{3,}\b", rewritten))

    # split whitelist into markers and wavelengths
    original_alpha = {w for w in original_whitelist if any(c.isalpha() for c in w)}
    original_alpha = {w for w in original_alpha if w.upper() == w} # keep only all caps
    original_num = {w for w in original_whitelist if w.isdigit()}

    # see if any new all caps
    extra_caps = set()
    for tok in new_tokens:
        if tok.isdigit(): # numeric token only ok if present in whitelist
            if tok not in original_num:
                extra_caps.add(tok) 

        else: # alphabetic token only ok if present in whitelist
            if tok not in original_alpha:
                extra_caps.add(tok)

    # return whether new caps were found and list of them
    return (len(extra_caps) > 0), extra_caps








# --- Generate Rewrites ---

# function to generate k rewrites for each entry
def generate_rewrites_for_entry(pipe, original_prompt, k, max_new_tokens, temperature, top_p):

    # build prompt
    prompts = [build_prompt(original_prompt) for _ in range(k)]

    # get eos and pad token ids
    eos_id = pipe.tokenizer.eos_token_id
    pad_id = pipe.tokenizer.pad_token_id or eos_id

    # generate rewrites
    outputs = pipe(
        prompts,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        num_return_sequences=1,
        return_full_text=True,
        batch_size=min(k, 4) # keep small batch for stability
    )

    # flatten list of lists when inputs is a list
    flat_outputs = []
    for out in outputs:
        if isinstance(out, list):
            flat_outputs.extend(out)
        else:
            flat_outputs.append(out)

    # extract rewritten texts
    rewritten_texts = []
    for out in flat_outputs:
        generated_text = out.get('generated_text') or out.get('text') or ""
        tail = generated_text.split("OUTPUT:")[-1]
        rewritten_texts.append(postprocess_text(tail))

    return rewritten_texts


# --- I/O Functions ---

# function to load prompts from json
def load_prompts_json(path):

    with open(path, 'r') as f:
        data = json.load(f)
    return data

# function to normalize keys
def normalize_key(key):

    return key.strip().lower().replace(" ", "_").replace("-", "_")

# function to turn a sentence into a list of tokens for ngram overlap checks
def _normalize_for_ngrams(text):

    # make lowercase and remove punctuation
    text = re.sub(r"[^\w\s-]", " ", text.lower()).strip()
    return re.findall(r"\w+(?:-\w+)?", text) # extract words and hyphenated words

# function to get ngram set from a string
def ngram_set(text, n):
    words = _normalize_for_ngrams(text) # tokenize
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)} # return n-grams

# function to compute jaccard similarity for 4-grams
def jaccard_4gram(a, b):
    A, B = ngram_set(a, 4), ngram_set(b, 4)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B) # return intersection over union

# function to expand and merge prompts
def expand_prompts(input_jsons, pipe, encoder, k, sim_thresh=0.80, max_new_tokens=96, 
                   temperature=0.2, top_p=0.9, banned_words=None, rw_self_sim_thresh=0.92, rw_jaccard_thresh=0.60):

    # create dict for merged prompts
    merged = {}

    # get banned words
    banned_words = banned_words or []

    # counters for keys, candidates, and accepted rewrites
    total_keys = 0
    total_candidates = 0
    total_accepted = 0

    # merge keys across files (allow later files to override earlier ones)
    for jp in input_jsons:
        raw = load_prompts_json(jp)
        for k_raw, v in raw.items():
            k_norm = normalize_key(k_raw)

            # support both legacy string prompts and dict with 'prompt' key
            if isinstance(v, dict) and 'orig' in v:
                merged[k_norm] = {'orig': v['orig'], 'rewrites': list(v.get('rewrites', []))}
            else:
                merged[k_norm] = {'orig': v, 'rewrites': []}

    # generate rewrites per key
    for key, payload in merged.items():

        # increment key counter
        total_keys += 1

        orig = payload['orig'].strip()

        # store detailed rejection reasons per rewrite
        payload.setdefault('rejection_reasons', [])

        # skip empty originals
        if not orig:
            continue

        # generate candidate rewrites
        cand_rewrites = generate_rewrites_for_entry(
            pipe, orig, k=k, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
        total_candidates += len(cand_rewrites) # increment candidate counter

        # lexical filtering
        whitelist = build_whitelist_from_original(orig)

        # loop over candidates and keep track of whether they pass prelim filtering or why they were rejected
        prelim_pass = []
        prelim_records = []
        
        for idx, rw in enumerate(cand_rewrites):

            # basic filtering
            ok, reasons = basic_filtering(orig, rw, banned_words)
            if not ok:
                payload['rejection_reasons'].append({'text': rw, 'rejection_reasons': reasons})
                print(f'[INFO] {key}: rewrite #{idx} rejected by basic filtering: {reasons}', flush=True)
                continue

            # check for new markers
            introduces, extras = introduces_new_markers(whitelist, rw)
            if introduces:
                reason = [f'introduced_new_marker:{m}' for m in extras]
                payload['rejection_reasons'].append({'text': rw, 'rejection_reasons': reason})
                print(f'[INFO] {key}: rewrite #{idx} rejected for introducing new markers: {extras}', flush=True)
                continue

            prelim_pass.append(rw)
            prelim_records.append({'idx': idx, 'text': rw})

        # after prelim filtering, apply similarity filtering
        if prelim_pass:
            sim_matrix = encoder.cosine_sim([orig], prelim_pass) # shape (1, n)
            similarities = sim_matrix[0].tolist() # get first row

            # map text to similarity for convenience
            text_to_sim = {record['text']: float(sim) for record, sim in zip(prelim_records, similarities)}

            filtered_rewrites = []
            for record, sim in zip(prelim_records, similarities):
                if float(sim) >= sim_thresh:
                    filtered_rewrites.append(record['text'])
                else:
                    payload['rejection_reasons'].append({
                        'text': record['text'],
                        'rejection_reasons': [f'low_similarity:{sim:.3f} < {sim_thresh:.3f}'],
                        'similarity': float(sim)
                    })
                    print(f'[INFO] {key}: rewrite #{record["idx"]} rejected for low similarity: {sim:.3f} < {sim_thresh:.3f}', flush=True)
        else:
            filtered_rewrites = []

        # deduplicate
        unique_rewrites = []
        seen = set()
        for rw in filtered_rewrites:
            key_rw = re.sub(r'\s+', ' ', rw.strip().lower())
            if key_rw not in seen:
                unique_rewrites.append(rw)
                seen.add(key_rw)
            else:
                payload['rejection_reasons'].append({'text': rw, 'rejection_reasons': ['duplicate_rewrite']})
                print(f'[INFO] {key}: duplicate rewrite rejected: {rw}', flush=True)

        # diversity gating among rewrites to keep only if dissimilar to those already kept
        diverse_rewrites = []
        for rw in unique_rewrites:
            keep = True

            # loop over already kept rewrites
            for kept in diverse_rewrites:

                # check cosine similarity (rewrite vs rewrite)
                cos_sim = float(encoder.cosine_sim([rw], [kept])[0, 0])

                # check jaccard 4-gram similarity (lexical overlap)
                jac = jaccard_4gram(rw, kept)
                if cos_sim >= rw_self_sim_thresh or jac > rw_jaccard_thresh:
                    keep = False
                    payload['rejection_reasons'].append({
                        'text': rw,
                        'rejection_reasons': [f'too_similar_to_other_rewrite: cos={cos_sim:.3f}, jacc={jac:.3f}']
                    })
                    print(f'[INFO] {key}: rewrite rejected for similarity to other rewrite: cos={cos_sim:.3f}, jacc={jac:.3f}', flush=True)
                    break
            if keep:
                diverse_rewrites.append(rw)

        # sort prelim records by similarity descending
        prelim_records_sorted = sorted(prelim_records, key=lambda r: text_to_sim[r['text']], reverse=True)

        # guarantee at least 2 rewrites by relaxing similarity if needed
        if prelim_pass and len(diverse_rewrites) < 2:

            for cand in prelim_records_sorted:

                text = cand['text']
                norm = re.sub(r'\s+', ' ', text.strip().lower())

                # avoid duplicates against anything already kept
                if any(re.sub(r'\s+', ' ', rw.strip().lower()) == norm for rw in diverse_rewrites):
                    continue

                # force add to final kept list
                diverse_rewrites.append(text)
                payload['rejection_reasons'].append({
                    'text': text,
                    'rejection_reasons': ['force_add_to_meet_minimum_rewrites'],
                    'similarity': float(text_to_sim.get(text, 0.0))
                })
                print(f'[INFO] {key}: added rewrite to meet minimum: {text}', flush=True)
                if len(diverse_rewrites) >= 2:
                    break

        # log how many rewrites kept
        num_accepted = len(diverse_rewrites)
        total_accepted += num_accepted # increment accepted counter
        num_total = len(cand_rewrites)
        print(f'[INFO] {key}: kept {num_accepted} / {num_total} rewrites after filtering', flush=True)

        # store final rewrites
        payload['rewrites'].extend(diverse_rewrites)

    print(f'[INFO] Processed {total_keys} keys: {total_accepted} accepted rewrites out of {total_candidates} candidates', flush=True)

    return merged


# --- Main ---

# main function
def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsons', nargs='+', required=True, help='One or more input JSON files with prompts to rewrite')
    parser.add_argument('--output_json', required=True, help='Single output JSON path for merged and expanded prompts')
    parser.add_argument('--llm_name', default='mistralai/Mistral-7B-Instruct-v0.2', help='HF model id or local path for instruction-tuned LLM')
    parser.add_argument('--device', default=None, help='"cuda" or "cpu" for LLM inference (default: auto-detect)')
    parser.add_argument('--k', type=int, default=6, help='Number of rewrites to generate per prompt')
    parser.add_argument('--sim_thresh', type=float, default=0.80, help='Cosine similarity threshold for filtering rewrites (original vs. rewrite)')
    parser.add_argument('--max_new_tokens', type=int, default=96, help='Maximum new tokens to generate for each rewrite')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature for LLM')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter for LLM')
    parser.add_argument('--batch_size', type=int, default=1, help='Unused; kept for compatibility')
    parser.add_argument('--rw_self_sim_thresh', type=float, default=0.92, help='Max cosine similarity between rewrites to consider them unique')
    parser.add_argument('--rw_jaccard_thresh', type=float, default=0.6, help='Max Jaccard similarity between rewrites to consider them unique')
    args = parser.parse_args()

    # get device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # build generation pipeline for LLM
    print(f'[INFO] Loading LLM: {args.llm_name} on device: {device}', flush=True)
    tok = AutoTokenizer.from_pretrained(args.llm_name)

    # ensure pad token for batching, use EOS if missing
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    
    # load model
    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.llm_name, 
                                                 torch_dtype=load_dtype,
                                                 low_cpu_mem_usage=True)
    
    if device == 'cuda':
        model = model.to('cuda')
    
    generation_pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tok,
        device=0 if device == 'cuda' else -1,
    )

    # build text encoder for similarity filtering
    print(f'[INFO] Loading text encoder: bert-base-uncased on device: {device}', flush=True)
    text_encoder = SimpleTextEncoder(encoder_name='bert-base-uncased', device=device)

    # merge and expand prompts
    print(f'[INFO] Expanding prompts from {len(args.input_jsons)} input files', flush=True)
    input_paths = [Path(p) for p in args.input_jsons]

    # show which files are being used
    print(f'[INFO] Input JSON files:', [str(p) for p in input_paths], flush=True)

    expanded_prompts = expand_prompts(
        input_jsons=input_paths,
        pipe=generation_pipe,
        encoder=text_encoder,
        k=args.k,
        sim_thresh=args.sim_thresh,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        banned_words=['beautiful', 'lovely', 'gorgeous', 'stunning', 'obvious', 'cause', 'caused by'],
        rw_self_sim_thresh=args.rw_self_sim_thresh,
        rw_jaccard_thresh=args.rw_jaccard_thresh,
    )

    # save to output json
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(expanded_prompts, f, indent=2, ensure_ascii=False)
    print(f'[INFO] Saved expanded prompts to {output_path}', flush=True)


# --- Main entry point ---

if __name__ == '__main__':
    main()
















    

