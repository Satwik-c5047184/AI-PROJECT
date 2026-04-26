"""
inference.py — ATE → ASC end-to-end inference on new text.

Usage:
    python inference.py "The battery life is great but the screen is dim."
    python inference.py --file reviews.txt
"""

import argparse
import sys
import torch
from transformers import (
    DistilBertForTokenClassification,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

ATE_MODEL_DIR = "models/ate"
ASC_MODEL_DIR = "models/asc"
MAX_LEN  = 128
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    """Load saved ATE and ASC models; raises FileNotFoundError if not trained yet."""
    import os
    for path in (ATE_MODEL_DIR, ASC_MODEL_DIR):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Model directory '{path}' not found — run main.py first to train."
            )

    ate_tokenizer = DistilBertTokenizerFast.from_pretrained(ATE_MODEL_DIR)
    ate_model = DistilBertForTokenClassification.from_pretrained(ATE_MODEL_DIR).to(DEVICE)
    ate_model.eval()

    asc_tokenizer = DistilBertTokenizerFast.from_pretrained(ASC_MODEL_DIR)
    asc_model = DistilBertForSequenceClassification.from_pretrained(ASC_MODEL_DIR).to(DEVICE)
    asc_model.eval()

    return ate_tokenizer, ate_model, asc_tokenizer, asc_model


# ── ATE ───────────────────────────────────────────────────────────────────────
def extract_aspects(sentence, ate_tokenizer, ate_model):
    """Return list of (aspect_text, word_start, word_end_exclusive) tuples."""
    words = sentence.split()
    if not words:
        return []

    enc = ate_tokenizer(
        words,
        is_split_into_words=True,
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt",
    )
    word_ids = enc.word_ids()

    with torch.no_grad():
        logits = ate_model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits  # (1, L, 3)

    preds = logits[0].argmax(-1).tolist()  # (L,)

    # Keep only the first subword prediction for each word
    word_labels = {}
    for subword_i, wi in enumerate(word_ids):
        if wi is not None and wi not in word_labels:
            word_labels[wi] = preds[subword_i]

    # Decode BIO sequence at word level
    spans, start = [], None
    for wi in sorted(word_labels):
        lbl = word_labels[wi]
        if lbl == 1:                            # B-ASP: start / restart span
            if start is not None:
                spans.append((start, wi))
            start = wi
        elif lbl == 2 and start is None:        # stray I-ASP — treat as B
            start = wi
        elif lbl == 0 and start is not None:    # O closes the open span
            spans.append((start, wi))
            start = None
    if start is not None:
        spans.append((start, len(word_labels)))

    return [(" ".join(words[s:e]), s, e) for s, e in spans]


# ── ASC ───────────────────────────────────────────────────────────────────────
def classify_sentiment(sentence, aspect, asc_tokenizer, asc_model):
    """Return predicted sentiment string for (sentence, aspect) pair."""
    enc = asc_tokenizer(
        sentence,
        aspect,
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = asc_model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits  # (1, 3)
    return ID2LABEL[logits[0].argmax(-1).item()]


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run(sentence, ate_tokenizer, ate_model, asc_tokenizer, asc_model):
    """ATE → ASC for one sentence.

    Returns list of dicts:
        [{"aspect": str, "start": int, "end": int, "sentiment": str}, ...]
    where start/end are word-level indices (end is exclusive).
    """
    results = []
    for aspect_text, start, end in extract_aspects(sentence, ate_tokenizer, ate_model):
        sentiment = classify_sentiment(sentence, aspect_text, asc_tokenizer, asc_model)
        results.append({"aspect": aspect_text, "start": start, "end": end, "sentiment": sentiment})
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
def _print_results(sentence, results):
    print(f"\nSentence: {sentence!r}")
    if not results:
        print("  (no aspects detected)")
        return
    for r in results:
        print(f"  [{r['sentiment']:>8}]  {r['aspect']!r}  (words {r['start']}–{r['end'] - 1})")


def main():
    parser = argparse.ArgumentParser(description="ABSA inference: ATE + ASC pipeline")
    parser.add_argument("sentence", nargs="?", help="Single sentence to analyse")
    parser.add_argument("--file", help="Text file with one sentence per line")
    args = parser.parse_args()

    if not args.sentence and not args.file:
        parser.print_help()
        sys.exit(1)

    print("Loading models...")
    try:
        ate_tok, ate_mdl, asc_tok, asc_mdl = load_models()
    except FileNotFoundError as e:
        sys.exit(str(e))

    print(f"  ATE: {ATE_MODEL_DIR}  |  ASC: {ASC_MODEL_DIR}  |  device: {DEVICE}")

    if args.sentence:
        results = run(args.sentence, ate_tok, ate_mdl, asc_tok, asc_mdl)
        _print_results(args.sentence, results)
    else:
        with open(args.file) as f:
            sentences = [line.strip() for line in f if line.strip()]
        print(f"  Processing {len(sentences)} sentence(s)...")
        for sentence in sentences:
            results = run(sentence, ate_tok, ate_mdl, asc_tok, asc_mdl)
            _print_results(sentence, results)


if __name__ == "__main__":
    main()
