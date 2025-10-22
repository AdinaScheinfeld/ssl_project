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
    "- stays concise (1-2 sentences)."
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
    RE_BAD = re.compile(r"\b(beautiful|lovely|gorgeous|stunning|obvious|clear|clearly)\b", re.IGNORECASE)

    # ensure not too short
    if len(rewritten) < 15:
        return False
    
    # ensure subjective adjectives not present
    if RE_BAD.search(rewritten):
        return False
    
    # ensure banned terms not present
    for b in banned_terms:
        if re.search(rf"\b{re.escape(b)}\b", rewritten, re.IGNORECASE):
            return False
        
    # avoid "unknown", "unannotated" if original is specific
    if ("Unknown" in rewritten or "unannotated" in rewritten.lower()) and len(original) > 40:
        return False

    # if all checks passed, return True
    return True

# function to build whitelist of terms from original prompts (ex: wavelengths, antibody)
def build_whitelist_from_original(original):

    # create whitelist
    whitelist = set()

    # wavelengths
    for m in re.findall(r"\b(4\d{2}|5\d{2}|6\d{2})\s*nm\b", original):
        whitelist.add(f'{m} nm')

    # antibodies/markers (simple heuristic of uppercase words >= 3 letters, ex: CTIP2, GFAP, LYVE1, DBH, etc)
    for m in re.findall(r"\b([A-Z0-9]{3,})\b", original):
        whitelist.add(m)

    return sorted(whitelist)

# function to reject rewrites that introduce new all caps marker-like tokens that were'nt in the original
def introduces_new_markers(original_whitelist, rewritten):

    # find all new and old caps
    new_caps = set(re.findall(r"\b[A-Z0-9]{3,}\b", rewritten))
    old_caps = set([w for w in original_whitelist if w.isupper()])

    # see if any new all caps
    extra_caps = new_caps - old_caps

    return len(extra_caps) > 0


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

# function to expand and merge prompts
def expand_prompts(input_jsons, pipe, encoder, k, sim_thresh=0.80, max_new_tokens=96, temperature=0.2, top_p=0.9, banned_words=None):

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

        # list of accepted rewrites
        accepted_rewrites = []

        # loop over candidates
        for rw in cand_rewrites:

            # basic filtering
            if not basic_filtering(orig, rw, banned_words):
                continue

            # check for new markers
            if introduces_new_markers(whitelist, rw):
                continue

            # keep if passes all filters
            accepted_rewrites.append(rw)

        # similarity filtering to keep diverse set
        if accepted_rewrites:
            similarities = encoder.cosine_sim([orig] * len(accepted_rewrites), accepted_rewrites).reshape(-1)
            filtered_rewrites = [rw for rw, s in zip(accepted_rewrites, similarities) if float(s) >= sim_thresh]

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

        # log how many rewrites kept
        num_accepted = len(unique_rewrites)
        total_accepted += num_accepted # increment accepted counter
        num_total = len(cand_rewrites)
        print(f'[INFO] {key}: kept {num_accepted} / {num_total} rewrites after filtering', flush=True)

        # store final rewrites
        payload['rewrites'].extend(unique_rewrites)

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
        banned_words=['beautiful', 'lovely', 'gorgeous', 'stunning', 'obvious', 'clear', 'clearly', 'cause', 'caused by']
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
















    

