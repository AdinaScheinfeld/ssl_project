# my_pretrain_utils.py - Utility functions for pre-training tasks
# Reconstructs pretrained text tower from a Lightning checkpoint 
# and exposes a callable text_encode_fn(prompts)->Tensor used by the inpainting module's CLIPTextEncoderWrapper

# --- Setup ---

# imports
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # disable tokenizer parallelism warnings

from monai.data.meta_tensor import MetaTensor

from transformers import AutoTokenizer, AutoModel
from transformers import logging as hf_logging
hf_logging.set_verbosity_error() # suppress HF warnings

import torch
import torch.nn as nn
from torch.serialization import add_safe_globals


# config
HF_NAME = 'bert-base-uncased'  # HuggingFace model name for text encoder
TEXT_OUT_DIM = 512 # final projected text embedding dimension used during pretraining (default/fallback if checkpoint doesn't contain proj weights)

# --- Text Backbone ---

# Text backbone (HF) and projection head with namees matching pretraining checkpoint
class _TextTower(nn.Module):

    # init
    def __init__(self, hf_name=HF_NAME, out_dim=TEXT_OUT_DIM):
        super().__init__()

        # load HF model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=False) # use_fast=False to avoid tokenization warnings
        self.text_encoder = AutoModel.from_pretrained(hf_name) # provides embeddings and encoder layer

        # get hidden size from HF model config
        hidden_size = self.text_encoder.config.hidden_size

        # mirror 2 layer mlp head + GELU (to match text_proj.* parameters in checkpoint)
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, out_dim, bias=True), # text_proj.0.{weight/bias}: (512,768)/(512,)
            nn.GELU(), # GELU (no weights)
            nn.Linear(out_dim, out_dim, bias=True), # text_proj.2.{weight/bias}: (512,512)/(512,)
            nn.LayerNorm(out_dim, elementwise_affine=True) # text_proj.3.{weight/bias}: (512,)/(512,)
        )

    # encode (tokenize -> HF encoder -> pool -> project, returns (B, out_dim))
    def encode(self, prompts, device):

        # tokenize prompts (B x T)
        self.eval().to(device)
        toks = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
        output = self.text_encoder(**toks) # HF model forward

        # prefer pooler output if available, else mean pool last hidden state
        if getattr(output, 'pooler_output', None) is not None:
            z = output.pooler_output # (B, hidden_size)
        else:
            z = output.last_hidden_state.mean(dim=1) # (B, hidden_size)
        z = self.text_proj(z) # (B, out_dim)
        return z
    

# function to strip prefixes
def _strip_prefix(key):
    for p in ('model.', 'module.', 'student_model.', 'student.', 'teacher_model.', 'net.', 'encoder.'):
        if key.startswith(p):
            return key[len(p):]
    return key

# function to infer output embedding dim from checkpoint
def _infer_text_out_dim_from_state(state_dict):

    # list of candidate projection weight keys
    candidates = [
        "text_proj.0.weight",
        "text_proj.2.weight",
        "text_proj.3.weight",
        "text_proj.0.bias",
        "text_proj.2.bias",
        "text_proj.3.bias",
    ]

    # search for candidates in state dict
    for key in candidates:
        v = state_dict.get(key, None)
        if isinstance(v, torch.Tensor):

            # weights are 2d, biases are 1d
            if v.ndim == 2:
                return int(v.shape[0]) # out_dim is first dim of weight
            
            # bias case
            elif v.ndim == 1:
                return int(v.shape[0]) # out_dim is size of bias
            
    # return default if no candidates found
    return TEXT_OUT_DIM


# function to build components from pretraining checkpoint
def build_components(ckpt_path, strict=False):

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load checkpoint and filter relevant keys
    add_safe_globals([MetaTensor]) # for loading monai MetaTensor if present
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = blob.get('state_dict', blob) # get state_dict from checkpoint blob if present

    # filter and strip prefixes
    filtered_state = {}
    for k, v in state.items():
        stripped_k = _strip_prefix(k)
        if stripped_k.startswith('text_encoder.') or stripped_k.startswith('text_proj.'): # keep only text tower keys
            filtered_state[stripped_k] = v

    # infer out_dim from checkpoint before instantiating text tower
    inferred_out_dim = _infer_text_out_dim_from_state(filtered_state)
    if inferred_out_dim != int(TEXT_OUT_DIM):
        print(f'[INFO] Inferred TEXT_OUT_DIM={inferred_out_dim} from checkpoint (overriding default {TEXT_OUT_DIM})', flush=True)
    else:
        print(f'[INFO] Using default TEXT_OUT_DIM={TEXT_OUT_DIM}', flush=True)

    # instantiate model with attribute names matching checkpoint prefixes
    text_tower = _TextTower(hf_name=HF_NAME, out_dim=inferred_out_dim)

    # load state dict
    incompat = text_tower.load_state_dict(filtered_state, strict=strict)
    missing = getattr(incompat, 'missing_keys', [])
    unexpected = getattr(incompat, 'unexpected_keys', [])
    print(f'[INFO] Loaded text tower from checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.', flush=True)
    if strict and (len(missing) > 0 or len(unexpected) > 0):
        raise RuntimeError('Strict load failed for text tower from checkpoint. Check HF_NAME/TEXT_OUT_DIM and key prefixes.\n'
                           f'Missing: {missing[:10]}...\nUnexpected: {unexpected[:10]}...')
    
    # expose callable used by inpainting module
    @torch.no_grad()
    def text_encode_fn(prompts):
        return text_tower.encode(prompts, device=device)
    
    # inpainting does not need image encoder from this utility
    def img_encode_fn(*_args, **_kwargs):
        raise NotImplementedError('Image encoder not implemented in my_pretrain_utils.build_components')
    
    return img_encode_fn, text_encode_fn, device
      
































