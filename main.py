import torch
from data   import load_raw, plot_eda, get_loaders
from ate    import train_ate, evaluate_ate
from asc    import train_asc, evaluate_asc
from graphs import plot_training_graphs

torch.manual_seed(42)


def main():
    print("=" * 60)
    print("  ABSA Customer Feedback Intelligence System")
    print("  SemEval-2014 Laptops  |  DistilBERT pipeline")
    print("=" * 60)

    # ── 1. Load data & EDA graphs ─────────────────────────────────
    print("\n[1/5] Loading dataset and generating EDA graphs...")
    train_df, test_df = load_raw()
    print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
    plot_eda(train_df, test_df)

    # ── 2. Build data loaders ─────────────────────────────────────
    print("\n[2/5] Building data loaders (80/20 sentence-level split)...")
    loaders = get_loaders(train_df, test_df)
    print(f"  ATE — train: {len(loaders['ate_train'])} batches  "
          f"val: {len(loaders['ate_val'])} batches  "
          f"test: {len(loaders['ate_test'])} batches")
    print(f"  ASC — train: {len(loaders['asc_train'])} batches  "
          f"val: {len(loaders['asc_val'])} batches  "
          f"test: {len(loaders['asc_test'])} batches")

    # ── 3. Train ATE ──────────────────────────────────────────────
    print("\n[3/5] Training ATE model (Aspect Term Extraction)...")
    ate_model, ate_history = train_ate(loaders)
    ate_results = evaluate_ate(ate_model, loaders["ate_test"])

    # ── 4. Train ASC ──────────────────────────────────────────────
    print("\n[4/5] Training ASC model (Aspect Sentiment Classification)...")
    asc_model, asc_history = train_asc(loaders)
    asc_results = evaluate_asc(asc_model, loaders["asc_test"])

    # ── 5. Generate post-training graphs ─────────────────────────
    print("\n[5/5] Generating training and evaluation graphs...")
    plot_training_graphs(ate_history, asc_history, asc_results)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  ATE  Precision : {ate_results['precision']:.4f}")
    print(f"  ATE  Recall    : {ate_results['recall']:.4f}")
    print(f"  ATE  Span F1   : {ate_results['f1']:.4f}")
    print(f"  ASC  Accuracy  : {asc_results['accuracy']:.4f}")
    print(f"  ASC  Macro-F1  : {asc_results['macro_f1']:.4f}")
    print(f"\n  All 10 graphs saved to 'graphs/'")
    print(f"  Best ATE model saved to 'models/ate/'")
    print(f"  Best ASC model saved to 'models/asc/'")
    print("=" * 60)


if __name__ == "__main__":
    main()
#
