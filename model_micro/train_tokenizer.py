"""Prune Qwen's tokenizer to keep ONLY tokens used in the dataset + NUM.

Strategy: Scan all splits, collect used token IDs, remap to a compact
vocabulary. This preserves pretrained embedding knowledge since each
new token ID maps directly to a Qwen embedding row.

Output:
  model_micro/micro_vocab.json          — pruned vocab {token_str: new_id}
  model_micro/micro_token_mapping.json  — old_id -> new_id for weight slicing
"""
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# Project root: one level up from this script (model_micro/ -> Thesis/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================================================================
# 1. Load Qwen tokenizer
# ====================================================================
tok = AutoTokenizer.from_pretrained(
    os.path.join(PROJECT_ROOT, 'model', 'qwen3.5-0.8b'), trust_remote_code=True
)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'nvidia_warehouse_dataset')
splits = {
    'train': os.path.join(DATA_DIR, 'train.json'),
    'val':   os.path.join(DATA_DIR, 'val.json'),
    'test':  os.path.join(DATA_DIR, 'test.json'),
}

# System prompt (must be tokenizable)
SYSTEM_PROMPT = (
    "You are a spatial reasoning assistant.\n"
    "Your response MUST be exactly one line in this format:\n"
    "CATEGORY | VALUE\n\n"
    "CATEGORY must be exactly one of these four words:\n"
    "  left_right\n  mcq\n  distance\n  count\n\n"
    "The separator ' | ' (space pipe space) is mandatory.\n\n"
    "VALUE rules:\n"
    '- left_right: "left" or "right" (with double quotes)\n'
    '- mcq: "0", "1", "2", etc. (with double quotes)\n'
    "- distance: NUM (the word NUM, Number Head predicts the value)\n"
    "- count: NUM (the word NUM, Number Head predicts the value)\n\n"
    "Examples:\n"
    'left_right | "left"\nmcq | "1"\ndistance | NUM\ncount | NUM\n\n'
    "Output ONLY the category, pipe, and value. Nothing else."
)

# ====================================================================
# 2. Scan ALL splits + ALL fields for used token IDs
# ====================================================================
all_ids = set()
per_split = {}

for name, path in splits.items():
    split_ids = set()
    data = json.load(open(path))
    for item in tqdm(data, desc=f"Scanning {name:>5}"):
        # All conversations (human + gpt)
        for conv in item.get('conversations', []):
            split_ids.update(tok.encode(conv.get('value', '')))

        # Category, normalized answer, freeform answer
        cat = item.get('category', '')
        if cat:
            split_ids.update(tok.encode(cat))
        norm_ans = item.get('normalized_answer', '')
        if norm_ans is not None:
            split_ids.update(tok.encode(str(norm_ans)))
        freeform = item.get('freeform_answer', '')
        if freeform:
            split_ids.update(tok.encode(freeform))

    all_ids.update(split_ids)
    per_split[name] = split_ids
    print(f"  {name:>5}: {len(data):>7} samples, {len(split_ids):>4} unique tokens")

# Add special tokens + system prompt
for sid in tok.all_special_ids:
    all_ids.add(sid)
all_ids.update(tok.encode(SYSTEM_PROMPT))

print(f"\n{'='*60}")
print(f"Total unique token IDs: {len(all_ids)}")
print(f"Original vocab size:    {tok.vocab_size}")

# Coverage check
test_only = per_split['test'] - per_split['train']
if test_only:
    print(f"\n  [!] TOKENS IN TEST BUT NOT IN TRAIN: {len(test_only)}")
    for tid in sorted(test_only):
        print(f"    ID {tid:>6}: '{tok.decode([tid])}'")
else:
    print(f"  [OK] All test tokens covered by train.")
# ====================================================================
# 3. Build compact vocab: old_id -> new_id (0..N-1)
#    ' NUM' (space-merged BPE) is naturally included from system prompt scan.
# ====================================================================
kept_old_ids = sorted(all_ids)  # sorted for reproducibility
total_vocab = len(kept_old_ids)

# old_id -> new_id mapping
old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}

# Find where ' NUM' (the actual context-dependent token) landed in the mapping.
# BPE is context-dependent: in 'distance | NUM', tokenizer produces ' NUM' (with space),
# NOT bare 'NUM'. We encode in context to get the correct old_id.
ctx_ids = tok.encode("| NUM", add_special_tokens=False)
NUM_OLD_ID = ctx_ids[-1]  # ' NUM' with leading space
bare_num_id = tok.encode("NUM", add_special_tokens=False)[0]  # bare 'NUM' (won't appear)

if NUM_OLD_ID in old_to_new:
    NUM_TOKEN_NEW_ID = old_to_new[NUM_OLD_ID]
    print(f"\n  [OK] NUM token: old_id={NUM_OLD_ID} (' NUM') -> new_id={NUM_TOKEN_NEW_ID}")
else:
    print(f"\n  [!] NUM token (old_id={NUM_OLD_ID}) NOT found in scanned tokens!")
    print(f"    Adding it manually...")
    kept_old_ids.append(NUM_OLD_ID)
    kept_old_ids.sort()
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}
    NUM_TOKEN_NEW_ID = old_to_new[NUM_OLD_ID]
    total_vocab = len(kept_old_ids)

# Note: bare 'NUM' (old={bare_num_id}) may also be in vocab from scan,
# but it's unused during training — only ' NUM' appears in context.
if bare_num_id in old_to_new:
    print(f"  ℹ Bare 'NUM' (old_id={bare_num_id}) also in vocab -> new_id={old_to_new[bare_num_id]} (unused)")

print(f"\n{'='*60}")
print(f"Kept tokens:     {total_vocab}")
print(f"NUM token:       new_id = {NUM_TOKEN_NEW_ID} (old_id = {NUM_OLD_ID}, context: ' NUM')")
print(f"Total vocab:     {total_vocab}")
print(f"Embedding size:  [{total_vocab}, 1024] = {total_vocab * 1024 / 1e6:.2f}M params")

# ====================================================================
# 4. Load original vocab.json for token strings
# ====================================================================
original_vocab = json.load(
    open(os.path.join(PROJECT_ROOT, 'model', 'qwen3.5-0.8b', 'vocab.json'))
)
id_to_token = {v: k for k, v in original_vocab.items()}

# For special tokens not in vocab.json (they're in added_tokens.json)
added_tokens_path = os.path.join(PROJECT_ROOT, 'model', 'qwen3.5-0.8b', 'added_tokens.json')
try:
    added = json.load(open(added_tokens_path))
    for token_str, token_id in added.items():
        id_to_token[token_id] = token_str
except FileNotFoundError:
    # Try tokenizer config instead
    for old_id in kept_old_ids:
        if old_id not in id_to_token:
            decoded = tok.decode([old_id])
            id_to_token[old_id] = decoded

# Build pruned vocab: {token_string: new_id}
pruned_vocab = {}
for old_id, new_id in sorted(old_to_new.items(), key=lambda x: x[1]):
    token_str = id_to_token.get(old_id, tok.decode([old_id]))
    pruned_vocab[token_str] = new_id

# ====================================================================
# 5. Save outputs
# ====================================================================
# vocab.json
vocab_path = os.path.join(SCRIPT_DIR, 'micro_vocab.json')
with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(pruned_vocab, f, ensure_ascii=False, indent=2)

# Token mapping (for create_micro.py to slice embeddings)
mapping_path = os.path.join(SCRIPT_DIR, 'micro_token_mapping.json')
with open(mapping_path, 'w') as f:
    json.dump({
        'kept_old_ids': kept_old_ids,
        'old_to_new': {str(k): v for k, v in old_to_new.items()},
        'num_token_id': NUM_TOKEN_NEW_ID,
        'total_vocab': total_vocab,
    }, f, indent=2)

print(f"\nSaved:")
print(f"  {vocab_path} ({len(pruned_vocab)} entries)")
print(f"  {mapping_path}")
print(f"  -> Use: new_embed = old_embed[kept_old_ids]  # [{total_vocab}, 1024]")
print(f"  -> NUM token uses pretrained Qwen embedding (no random init needed)")

# ====================================================================
# 6. Display full vocabulary
# ====================================================================
print(f"\n{'='*60}")
print(f"FULL PRUNED VOCABULARY ({total_vocab} tokens)")
print(f"{'='*60}")
for token_str, new_id in sorted(pruned_vocab.items(), key=lambda x: x[1]):
    old_id = kept_old_ids[new_id] if new_id < len(kept_old_ids) else "NEW"
    print(f"  {new_id:>3} (old:{str(old_id):>7}): {repr(token_str)}")

# ====================================================================
# 7. Test: verify round-trip works
# ====================================================================
print(f"\n{'='*60}")
print("Test: tokenize with Qwen -> remap to new IDs")
test_texts = [
    'distance | NUM',
    'left_right | "left"',
    'mcq | "2"',
    'count | 3',
    'Which pallet from <mask> <mask> is the closest to the transporter at <mask>?',
]
all_covered = True
for text in test_texts:
    old_ids = tok.encode(text)
    new_ids = []
    missing = []
    for oid in old_ids:
        if oid in old_to_new:
            new_ids.append(old_to_new[oid])
        else:
            new_ids.append(-1)
            missing.append(oid)
            all_covered = False
    
    tokens = [tok.decode([oid]) for oid in old_ids]
    print(f"\n  '{text}'")
    print(f"    old_ids:  {old_ids}")
    print(f"    new_ids:  {new_ids}")
    print(f"    tokens:   {tokens}")
    if missing:
        print(f"    [!] MISSING: {missing} -> {[tok.decode([m]) for m in missing]}")

if all_covered:
    print(f"\n  [OK] All test tokens covered by pruned vocab!")
