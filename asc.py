"""
asc.py  —  Aspect Sentiment Classification: DistilBERT sequence classifier
           Input:  [CLS] sentence [SEP] aspect [SEP]
           Output: negative (0) / neutral (1) / positive (2)
"""

import os
import torch
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from data import MODEL_NAME, NUM_EPOCHS, tokenizer

ASC_SAVE_DIR = "models/asc"

DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SENTIMENT_LABELS = 3   # negative, neutral, positive


# ── Shared eval loop ─────────────────────────────────────────────────────────
def _run_eval(model, loader):
    model.eval()
    total_loss, all_preds, all_gold = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            total_loss  += out.loss.item()
            all_preds.extend(out.logits.argmax(-1).tolist())
            all_gold.extend(batch["labels"].tolist())
    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_gold, all_preds)
    macro_f1 = f1_score(all_gold, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, all_gold, all_preds


# ── Training ─────────────────────────────────────────────────────────────────
def train_asc(loaders, num_epochs=NUM_EPOCHS, lr=2e-5):
    """Fine-tune DistilBERT for sentiment classification; return (model, history)."""
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_SENTIMENT_LABELS
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    history   = []
    best_val_f1 = -1.0

    os.makedirs(ASC_SAVE_DIR, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loaders["asc_train"], desc=f"ASC Epoch {epoch}/{num_epochs}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()

        train_loss = total_loss / len(loaders["asc_train"])
        val_loss, val_acc, val_f1, _, _ = _run_eval(model, loaders["asc_val"])
        history.append({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

        saved = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(ASC_SAVE_DIR)
            tokenizer.save_pretrained(ASC_SAVE_DIR)
            saved = "  [saved]"

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}{saved}")

    return model, history


# ── Test evaluation ───────────────────────────────────────────────────────────
def evaluate_asc(model, loader):
    """Evaluate on test set; print metrics and return results dict."""
    _, acc, macro_f1, gold, preds = _run_eval(model, loader)
    cm = confusion_matrix(gold, preds)
    print(f"\nASC Test  —  Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}")
    return {
        "accuracy":         acc,
        "macro_f1":         macro_f1,
        "confusion_matrix": cm.tolist(),
        "gold":             gold,
        "preds":            preds,
    }
