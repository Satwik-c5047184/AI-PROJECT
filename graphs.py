import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data import GRAPHS_DIR, ID2LABEL


def plot_training_graphs(ate_history, asc_history, asc_eval):
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    ate_epochs = [h["epoch"] for h in ate_history]
    asc_epochs = [h["epoch"] for h in asc_history]

    # Graph 6: ATE train vs validation loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ate_epochs, [h["train_loss"] for h in ate_history],
            "o-", label="Train Loss", color="#2980b9")
    ax.plot(ate_epochs, [h["val_loss"] for h in ate_history],
            "s--", label="Val Loss", color="#e74c3c")
    ax.set_title("ATE Training & Validation Loss per Epoch", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_xticks(ate_epochs); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/6_ate_loss.png", dpi=150); plt.close()

    # Graph 7: ATE validation span F1 per epoch
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ate_epochs, [h["val_f1"] for h in ate_history],
            "o-", color="#27ae60", label="Val Span F1")
    ax.fill_between(ate_epochs, [h["val_precision"] for h in ate_history],
                    [h["val_recall"] for h in ate_history], alpha=0.15, color="#27ae60",
                    label="Precision–Recall band")
    ax.set_title("ATE Validation Span F1 per Epoch", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_xticks(ate_epochs); ax.set_ylim(0, 1); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/7_ate_f1.png", dpi=150); plt.close()

    # Graph 8: ASC train vs validation loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(asc_epochs, [h["train_loss"] for h in asc_history],
            "o-", label="Train Loss", color="#2980b9")
    ax.plot(asc_epochs, [h["val_loss"] for h in asc_history],
            "s--", label="Val Loss", color="#e74c3c")
    ax.set_title("ASC Training & Validation Loss per Epoch", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_xticks(asc_epochs); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/8_asc_loss.png", dpi=150); plt.close()

    # Graph 9: ASC validation accuracy and Macro-F1 per epoch
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(asc_epochs, [h["val_accuracy"] for h in asc_history],
            "o-", label="Val Accuracy", color="#8e44ad")
    ax.plot(asc_epochs, [h["val_macro_f1"] for h in asc_history],
            "s--", label="Val Macro-F1", color="#e67e22")
    ax.set_title("ASC Validation Metrics per Epoch", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_xticks(asc_epochs); ax.set_ylim(0, 1); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/9_asc_metrics.png", dpi=150); plt.close()

    # Graph 10: ASC confusion matrix on test set
    cm   = np.array(asc_eval["confusion_matrix"])
    lbls = [ID2LABEL[i] for i in range(3)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=lbls, yticklabels=lbls, ax=ax, linewidths=0.5)
    ax.set_title("ASC Confusion Matrix — Test Set", fontsize=13)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/10_asc_confusion_matrix.png", dpi=150); plt.close()

    print(f"  Training/evaluation graphs 6-10 saved to '{GRAPHS_DIR}/'")
