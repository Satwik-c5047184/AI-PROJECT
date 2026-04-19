"""
ate.py  —  Aspect Term Extraction: DistilBERT token classifier (BIO tagging)
"""

import torch
from transformers import DistilBertForTokenClassification
from torch.optim import AdamW
from tqdm import tqdm
from data import MODEL_NAME, NUM_EPOCHS

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BIO_LABELS = 3   # 0=O, 1=B-ASP, 2=I-ASP


# ── Span extraction & F1 ─────────────────────────────────────────────────────
def _extract_spans(labels):
    """Convert a BIO label list to a set of (start, end) token-index spans."""
    spans, start = [], None
    for i, lbl in enumerate(labels):
        if lbl == 1:                               # B-ASP: start new span
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif lbl == 2 and start is None:           # I-ASP without B-ASP
            start = i
        elif lbl not in (1, 2) and start is not None:  # O: close span
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(labels) - 1))
    return set(spans)


def _span_f1(gold_seqs, pred_seqs):
    tp = fp = fn = 0
    for gold, pred in zip(gold_seqs, pred_seqs):
        g = _extract_spans(gold)
        p = _extract_spans(pred)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# ── Shared eval loop ─────────────────────────────────────────────────────────
def _run_eval(model, loader):
    model.eval()
    total_loss, gold_seqs, pred_seqs = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            total_loss += out.loss.item()
            preds = out.logits.argmax(-1)   # (B, L)
            gold  = batch["labels"]         # (B, L)
            for p_row, g_row in zip(preds, gold):
                keep = g_row != -100        # exclude [CLS], [SEP], PAD
                pred_seqs.append(p_row[keep].tolist())
                gold_seqs.append(g_row[keep].tolist())
    avg_loss = total_loss / len(loader)
    prec, rec, f1 = _span_f1(gold_seqs, pred_seqs)
    return avg_loss, prec, rec, f1


# ── Training ─────────────────────────────────────────────────────────────────
def train_ate(loaders, num_epochs=NUM_EPOCHS, lr=2e-5):
    """Fine-tune DistilBERT for BIO token classification; return (model, history)."""
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_BIO_LABELS
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    history   = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loaders["ate_train"], desc=f"ATE Epoch {epoch}/{num_epochs}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()

        train_loss = total_loss / len(loaders["ate_train"])
        val_loss, val_p, val_r, val_f1 = _run_eval(model, loaders["ate_val"])
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_precision": val_p,
            "val_recall":    val_r,
            "val_f1":        val_f1,
        })
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_P={val_p:.4f}  val_R={val_r:.4f}  val_F1={val_f1:.4f}")

    return model, history


# ── Test evaluation ───────────────────────────────────────────────────────────
def evaluate_ate(model, loader):
    """Evaluate on test set; print and return span-level P/R/F1."""
    _, prec, rec, f1 = _run_eval(model, loader)
    print(f"\nATE Test  —  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    return {"precision": prec, "recall": rec, "f1": f1}
