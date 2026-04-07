"""Analyze questions from a dataset JSON to find every unique pattern.

Usage:
  python test/analyze_full_train.py                          # default: train.json
  python test/analyze_full_train.py --input data/nvidia_warehouse_dataset/val.json --output test/val_analysis.txt
  python test/analyze_full_train.py --input data/nvidia_warehouse_dataset/test.json --output test/test_analysis.txt
"""
import json, re, sys, argparse
from collections import defaultdict, Counter
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Analyze dataset questions for unique patterns.")
parser.add_argument("--input", "-i", default="data/nvidia_warehouse_dataset/train.json",
                    help="Path to input JSON file (default: train.json)")
parser.add_argument("--output", "-o", default=None,
                    help="Path to output analysis file (default: auto-generated from input name)")
args = parser.parse_args()

# Auto-generate output name from input if not specified
if args.output is None:
    import os
    base = os.path.splitext(os.path.basename(args.input))[0]
    args.output = f"analysis/{base}_analysis.txt"

print(f"Loading {args.input} ...")
data = json.load(open(args.input))
total = len(data)
print(f"Total samples: {total}")

OUT = args.output

# ─── helpers ───────────────────────────────────────────────────────────────────

def normalize_question(q):
    """Replace mask sequences and specific objects to get the template."""
    q = q.replace("<image>\n", "").strip()
    q = re.sub(r'(\s*<mask>)+', ' [MASKS]', q)
    q = re.sub(r'the pallet \[MASKS\]', 'OBJ', q)
    q = re.sub(r'pallet \[MASKS\]', 'OBJ', q)
    q = re.sub(r'the buffer region \[MASKS\]', 'OBJ', q)
    q = re.sub(r'buffer region \[MASKS\]', 'OBJ', q)
    q = re.sub(r'the transporter \[MASKS\]', 'OBJ', q)
    q = re.sub(r'transporter \[MASKS\]', 'OBJ', q)
    q = re.sub(r'the shelf \[MASKS\]', 'OBJ', q)
    q = re.sub(r'shelf \[MASKS\]', 'OBJ', q)
    q = re.sub(r'the shelves \[MASKS\]', 'OBJ', q)
    q = re.sub(r'shelves \[MASKS\]', 'OBJ', q)
    return q.strip()

KNOWN_KEYWORDS = {
    "left_right": ["left", "right"],
    "distance":   ["distance", "far", "close", "distant", "measure", "apart"],
    "count":      ["how many", "count", "number"],
    "mcq":        ["leftmost", "rightmost", "left", "right", "nearest", "closest",
                   "best", "suitable", "pick up", "automated", "convenient",
                   "optimal", "accessible", "idle", "retrieve", "assigned",
                   "smallest distance"],
}

ALL_KEYWORDS = [
    "leftmost", "rightmost", "left", "right",
    "nearest", "closest", "distance", "far", "close", "distant",
    "how many", "count", "number",
    "empty transporter", "automated", "pick up", "suitable",
    "optimal", "accessible", "idle", "retrieve", "assigned",
    "buffer", "shelf", "shelves", "transporter", "middle",
    "smallest distance", "convenient",
]

# ─── category inference (for datasets without category field) ─────────────────

CATEGORY_KEYWORDS = {
    "count":      ["how many", "count of", "number of", "total number"],
    "distance":   ["distance", "how far", "how close", "how distant", "measure", "apart"],
    "left_right": ["to the left", "to the right", "left of", "right of",
                   "left side", "right side", "left-hand", "right-hand",
                   "left or right", "positioned to the left", "positioned to the right"],
}

def infer_category(q_lower):
    """Infer category from question text using keywords (order matters)."""
    for cat in ["count", "distance", "left_right"]:
        if any(kw in q_lower for kw in CATEGORY_KEYWORDS[cat]):
            return cat
    return "mcq"  # default fallback

# ─── detect dataset type ──────────────────────────────────────────────────────

has_category = "category" in data[0]
has_answer = "normalized_answer" in data[0]
print(f"  has 'category' field: {has_category}")
print(f"  has 'normalized_answer' field: {has_answer}")
if not has_category:
    print("  -> Will infer category from question keywords")

# ─── process all samples ──────────────────────────────────────────────────────

templates = defaultdict(lambda: Counter())
keyword_counts = defaultdict(lambda: Counter())
answer_stats = defaultdict(list)
anomaly_list = []
mask_counts = defaultdict(list)
inferred_cats = Counter()

print("Analyzing all samples ...")
for item in tqdm(data, desc="Processing", ncols=80):
    q_raw = item["conversations"][0]["value"].replace("<image>\n", "").strip()
    q_norm = normalize_question(item["conversations"][0]["value"])
    q_lower = q_raw.lower()

    if has_category:
        cat = item["category"]
    else:
        cat = infer_category(q_lower)
        inferred_cats[cat] += 1

    templates[cat][q_norm] += 1
    if has_answer:
        answer_stats[cat].append(item["normalized_answer"])
    mask_counts[cat].append(q_raw.count("<mask>"))

    # Keyword counting
    for kw in ALL_KEYWORDS:
        if kw in q_lower:
            keyword_counts[cat][kw] += 1

    # Anomaly check
    is_known = any(kw in q_lower for kw in KNOWN_KEYWORDS.get(cat, []))
    if not is_known:
        ans = item.get("normalized_answer", "N/A")
        anomaly_list.append((cat, ans, q_raw))

# ─── write results ────────────────────────────────────────────────────────────

print(f"Writing results to {OUT} ...")
with open(OUT, "w") as f:
    def p(s=""):
        f.write(s + "\n")

    src = args.input.split("/")[-1]
    p(f"ANALYSIS OF {src.upper()} -- {total} samples")
    if not has_category:
        p(f"  NOTE: category was INFERRED from keywords (not in dataset)")
    if not has_answer:
        p(f"  NOTE: normalized_answer not present in dataset")
    p(f"{'='*110}")

    # ── Category distribution ──
    cats_found = sorted(templates.keys())
    p(f"\n{'='*110}")
    cat_label = "CATEGORY DISTRIBUTION (inferred)" if not has_category else "CATEGORY DISTRIBUTION"
    p(f"  {cat_label} ({total} samples)")
    p(f"{'='*110}")
    for cat in ["mcq", "left_right", "distance", "count"]:
        n = sum(templates[cat].values())
        if n > 0:
            p(f"  {cat:12s}: {n:6d} samples ({100*n/total:.1f}%)")
    p(f"  {'TOTAL':12s}: {total:6d}")

    # ── Answer stats per category (only if answers exist) ──
    if has_answer:
        p(f"\n{'='*110}")
        p(f"  ANSWER STATISTICS")
        p(f"{'='*110}")
        for cat in ["mcq", "left_right", "distance", "count"]:
            answers = answer_stats[cat]
            if not answers:
                continue
            ans_counter = Counter(str(a) for a in answers)
            masks = mask_counts[cat]
            p(f"\n  {cat.upper()} ({len(answers)} samples):")
            p(f"    Masks per question: min={min(masks)}, max={max(masks)}, avg={sum(masks)/len(masks):.1f}")
            p(f"    Unique answers: {len(ans_counter)}")
            p(f"    Top 20 answers:")
            for ans, cnt in ans_counter.most_common(20):
                p(f"      {ans:15s}: {cnt:5d} ({100*cnt/len(answers):.1f}%)")
    else:
        # Still show mask stats even without answers
        p(f"\n{'='*110}")
        p(f"  MASK STATISTICS (no answers available)")
        p(f"{'='*110}")
        for cat in ["mcq", "left_right", "distance", "count"]:
            masks = mask_counts[cat]
            if not masks:
                continue
            p(f"\n  {cat.upper()} ({len(masks)} samples):")
            p(f"    Masks per question: min={min(masks)}, max={max(masks)}, avg={sum(masks)/len(masks):.1f}")

    # ── Templates per category ──
    for cat in ["mcq", "left_right", "distance", "count"]:
        n_samples = sum(templates[cat].values())
        if n_samples == 0:
            continue
        sorted_t = templates[cat].most_common()
        p(f"\n{'='*110}")
        p(f"  {cat.upper()}  ({n_samples} samples, {len(sorted_t)} unique templates)")
        p(f"{'='*110}")

        for i, (tmpl, count) in enumerate(sorted_t):
            pct = 100 * count / n_samples
            p(f"\n  Template {i+1} ({count}x, {pct:.1f}%):")
            p(f"    {tmpl}")

        # Keyword frequency
        p(f"\n  {'─'*106}")
        p(f"  Keyword frequency:")
        kw_sorted = sorted(keyword_counts[cat].items(), key=lambda x: -x[1])
        for kw, cnt in kw_sorted:
            p(f"    {kw:25s}: {cnt:6d} ({100*cnt/n_samples:.1f}%)")

    # ── Anomalies ──
    p(f"\n{'='*110}")
    p(f"  ANOMALY CHECK: Questions that DON'T match known keyword patterns")
    p(f"{'='*110}")
    p(f"  Total anomalies: {len(anomaly_list)}/{total} ({100*len(anomaly_list)/total:.2f}%)")

    if anomaly_list:
        # Show unique anomaly templates
        anomaly_templates = Counter()
        for cat, ans, q in anomaly_list:
            anomaly_templates[(cat, normalize_question(q))] += 1

        p(f"  Unique anomaly templates: {len(anomaly_templates)}")
        p()
        for (cat, tmpl), cnt in anomaly_templates.most_common(50):
            p(f"  [{cat}] ({cnt}x): {tmpl}")
    else:
        p("  None found! All questions match known keyword patterns.")

print(f"Done! Results saved to {OUT}")
print(f"Anomalies: {len(anomaly_list)}/{total} ({100*len(anomaly_list)/total:.2f}%)")

