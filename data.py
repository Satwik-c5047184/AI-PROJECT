import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 16
NUM_EPOCHS  = 3
GRAPHS_DIR  = "graphs"

POLARITY2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL    = {0: "negative", 1: "neutral", 2: "positive"}

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


# ── Raw data ──────────────────────────────────────────────────────────────────
def load_raw():
    def read_domain(domain, split):
        with open(f"data/{domain}/{split}.json") as f:
            data = json.load(f)
        rows = []
        for sent in data:
            sentence = " ".join(sent["token"])
            for asp in sent["aspects"]:
                if asp["polarity"] not in POLARITY2ID:   # skip rare "conflict" labels
                    continue
                rows.append({
                    "tokens":   sent["token"],            # list of words (pre-tokenised)
                    "sentence": sentence,                 # full sentence string
                    "aspect":   " ".join(asp["term"]),    # aspect string
                    "asp_from": asp["from"],              # word-level start (inclusive)
                    "asp_to":   asp["to"],                # word-level end   (exclusive)
                    "polarity": asp["polarity"],
                    "domain":   domain,
                })
        return rows

    train_rows = read_domain("Laptops", "train") + read_domain("Restaurants", "train")
    test_rows  = read_domain("Laptops", "test")  + read_domain("Restaurants", "test")
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


# ── EDA Graphs 1–5 ────────────────────────────────────────────────────────────
def plot_eda(train_df, test_df):
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Graph 1: Sentiment distribution by domain (grouped bar)
    counts = (df.groupby(["domain", "polarity"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["positive", "neutral", "negative"], fill_value=0))
    counts.plot(kind="bar", figsize=(8, 5),
                color=["#2ecc71", "#95a5a6", "#e74c3c"], edgecolor="white")
    plt.title("Sentiment Distribution by Domain — SemEval-2014", fontsize=13)
    plt.xlabel("Domain"); plt.ylabel("Count")
    plt.xticks(rotation=0); plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/1_sentiment_distribution.png", dpi=150); plt.close()

    # Graph 2: Top 20 aspect terms (combined)
    top20 = Counter(df["aspect"].str.lower()).most_common(20)
    aspects, freqs = zip(*top20)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(aspects[::-1], freqs[::-1], color="steelblue")
    ax.set_title("Top 20 Most Frequent Aspect Terms (Combined)", fontsize=13)
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/2_top_aspects.png", dpi=150); plt.close()

    # Graph 3: Review sentence length distribution by domain (overlaid)
    df["words"] = df["sentence"].apply(lambda t: len(t.split()))
    fig, ax = plt.subplots(figsize=(8, 4))
    for domain, color in [("Laptops", "steelblue"), ("Restaurants", "coral")]:
        sub = df[df["domain"] == domain]["words"]
        ax.hist(sub, bins=30, alpha=0.6, color=color, edgecolor="white",
                label=f"{domain}  (mean {sub.mean():.1f})")
    ax.set_title("Review Sentence Length Distribution by Domain", fontsize=13)
    ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/3_review_lengths.png", dpi=150); plt.close()

    # Graph 4: Number of aspects per sentence — side-by-side per domain
    asp_per_sent = df.groupby(["domain", "sentence"]).size().reset_index(name="count")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (domain, color) in zip(axes, [("Laptops", "steelblue"), ("Restaurants", "coral")]):
        vals = asp_per_sent[asp_per_sent["domain"] == domain]["count"]
        ax.hist(vals, bins=range(1, int(vals.max()) + 2),
                color=color, edgecolor="white", align="left")
        ax.set_title(f"{domain}", fontsize=11)
        ax.set_xlabel("Aspects per Sentence"); ax.set_ylabel("Sentences")
        ax.set_xticks(range(1, int(vals.max()) + 1))
    plt.suptitle("Number of Aspects per Sentence by Domain", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/4_aspects_per_sentence.png", dpi=150); plt.close()

    # Graph 5: Sentiment for top-5 aspects (combined)
    top5 = [a for a, _ in Counter(df["aspect"].str.lower()).most_common(5)]
    df["aspect_lower"] = df["aspect"].str.lower()
    df_top = df[df["aspect_lower"].isin(top5)]
    pivot = df_top.pivot_table(index="aspect_lower", columns="polarity",
                               aggfunc="size", fill_value=0)
    for col in ["positive", "neutral", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot[["positive", "neutral", "negative"]].plot(
        kind="bar", figsize=(9, 5), color=["#2ecc71", "#95a5a6", "#e74c3c"])
    plt.title("Sentiment Distribution for Top 5 Aspects", fontsize=13)
    plt.xlabel("Aspect Term"); plt.ylabel("Count")
    plt.xticks(rotation=15); plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/5_sentiment_per_aspect.png", dpi=150); plt.close()

    print(f"  EDA graphs 1-5 saved to '{GRAPHS_DIR}/'")


# ── BIO label builder ─────────────────────────────────────────────────────────
def _build_bio(tokens, aspect_spans):
    """Tokenise a pre-tokenized sentence with word-level BIO labels.

    Only the first subword of each word gets a real label; continuations are
    set to -100 so they are ignored by CrossEntropyLoss.
    aspect_spans: list of (from, to) word-index pairs, 'to' is exclusive.
    """
    enc = tokenizer(
        tokens, is_split_into_words=True,
        max_length=MAX_LEN, truncation=True, padding="max_length",
    )
    word_ids = enc.word_ids()
    labels, prev_wi = [], None
    for wi in word_ids:
        if wi is None:                          # [CLS], [SEP], or PAD
            labels.append(-100)
        elif wi != prev_wi:                     # first subword of this word
            lbl = 0                             # O by default
            for a_s, a_e in aspect_spans:
                if a_s <= wi < a_e:             # word is inside an aspect span
                    lbl = 1 if wi == a_s else 2  # B-ASP or I-ASP
                    break
            labels.append(lbl)
        else:                                   # continuation subword — ignore
            labels.append(-100)
        prev_wi = wi
    return enc["input_ids"], enc["attention_mask"], labels


# ── PyTorch Datasets ──────────────────────────────────────────────────────────
class ATEDataset(Dataset):
    """One sample per unique sentence; labels are BIO token tags (O/B-ASP/I-ASP)."""
    def __init__(self, df):
        self.items = []
        for sentence, grp in df.groupby("sentence"):
            tokens = grp.iloc[0]["tokens"]
            spans  = list(zip(grp["asp_from"], grp["asp_to"]))
            ids, mask, bio = _build_bio(tokens, spans)
            self.items.append({
                "input_ids":      torch.tensor(ids,  dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "labels":         torch.tensor(bio,  dtype=torch.long),
            })

    def __len__(self):         return len(self.items)
    def __getitem__(self, i):  return self.items[i]


class ASCDataset(Dataset):
    """One sample per (sentence, aspect) pair; label is sentiment class (0/1/2)."""
    def __init__(self, df):
        enc = tokenizer(
            df["sentence"].tolist(), df["aspect"].tolist(),
            max_length=MAX_LEN, truncation=True, padding="max_length",
        )
        self.input_ids      = torch.tensor(enc["input_ids"],      dtype=torch.long)
        self.attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        self.labels         = torch.tensor(
            [POLARITY2ID[p] for p in df["polarity"]], dtype=torch.long)

    def __len__(self):        return len(self.labels)
    def __getitem__(self, i): return {
        "input_ids":      self.input_ids[i],
        "attention_mask": self.attention_mask[i],
        "labels":         self.labels[i],
    }


# ── DataLoaders with sentence-level train/val split ───────────────────────────
def get_loaders(train_df, test_df):
    """80/20 sentence-level split to prevent leakage; returns dict of DataLoaders."""
    rng    = np.random.default_rng(42)
    unique = train_df["sentence"].unique().copy()
    rng.shuffle(unique)
    cut    = int(len(unique) * 0.8)
    tr_df  = train_df[train_df["sentence"].isin(unique[:cut])].reset_index(drop=True)
    val_df = train_df[train_df["sentence"].isin(unique[cut:])].reset_index(drop=True)

    def dl(ds, shuffle=False):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return {
        "ate_train": dl(ATEDataset(tr_df),   shuffle=True),
        "ate_val":   dl(ATEDataset(val_df)),
        "ate_test":  dl(ATEDataset(test_df)),
        "asc_train": dl(ASCDataset(tr_df),   shuffle=True),
        "asc_val":   dl(ASCDataset(val_df)),
        "asc_test":  dl(ASCDataset(test_df)),
    }
