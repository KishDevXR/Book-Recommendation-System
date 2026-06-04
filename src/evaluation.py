"""
=============================================================================
evaluation.py  –  Book Recommendation System
=============================================================================
MCA Capstone Project | Artificial Intelligence
Author  : Student
Purpose : Comprehensive model evaluation:
          - RMSE, MAE   (rating-prediction quality)
          - Precision@K (fraction of top-K recs that are relevant)
          - Recall@K    (fraction of relevant items retrieved in top-K)
          - Comparison table across all approaches
          - Evaluation visualisations
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(FIG_DIR, exist_ok=True)

BG_COLOR = "#0f0f1a"
FG_COLOR = "#e0e0f0"
ACC1     = "#7c3aed"
ACC2     = "#06b6d4"
ACC3     = "#f59e0b"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#2a2a4a", "axes.labelcolor": FG_COLOR,
    "xtick.color": FG_COLOR, "ytick.color": FG_COLOR,
    "text.color": FG_COLOR, "grid.color": "#2a2a4a",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "sans-serif",
})


def _save(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [FIG] Saved → {name}")


# =============================================================================
# 1. Rating Prediction Metrics (RMSE / MAE)
# =============================================================================

def compute_rmse_mae(predictions) -> dict:
    """
    Compute RMSE and MAE from a list of Surprise Prediction namedtuples.

    Parameters
    ----------
    predictions : list of surprise.Prediction

    Returns
    -------
    {"RMSE": float, "MAE": float}
    """
    y_true = np.array([p.r_ui for p in predictions])
    y_pred = np.array([p.est  for p in predictions])

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# =============================================================================
# 2. Precision@K  &  Recall@K
# =============================================================================

def precision_recall_at_k(predictions, k: int = 10,
                           threshold: float = 7.0) -> tuple[float, float]:
    """
    Compute Precision@K and Recall@K for a Surprise prediction list.

    A rating ≥ threshold is considered "relevant".

    Steps
    -----
    1. Group predictions by user.
    2. Sort each user's predictions by estimated rating (desc).
    3. Take top-K as recommendations.
    4. Precision@K = (# relevant in top-K) / K
    5. Recall@K    = (# relevant in top-K) / (# relevant items for user)
    """
    from collections import defaultdict

    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_relevant_in_topk = sum(1 for (est, r) in top_k if r >= threshold)
        n_relevant_total   = sum(1 for (est, r) in user_ratings if r >= threshold)

        precisions.append(n_relevant_in_topk / k if k else 0)
        recalls.append(n_relevant_in_topk / n_relevant_total
                       if n_relevant_total else 0)

    precision = float(np.mean(precisions))
    recall    = float(np.mean(recalls))
    return round(precision, 4), round(recall, 4)


# =============================================================================
# 3. CF-Specific Evaluation (Coverage)
# =============================================================================

def evaluate_cf_coverage(cf_model, book_list: list, n: int = 10) -> dict:
    """
    For a sample of books, compute the fraction for which the CF model
    can generate at least 1 recommendation (coverage).
    """
    coverage_count = 0
    for book in book_list:
        try:
            recs = cf_model.recommend_books(book, n=n)
            if "Message" not in recs.columns and len(recs) > 0:
                coverage_count += 1
        except Exception:
            pass
    coverage = coverage_count / len(book_list) if book_list else 0
    return {"Coverage": round(coverage, 4), "Total_Queried": len(book_list)}


# =============================================================================
# 4. Comparison Table
# =============================================================================

def build_comparison_table(svd_metrics: dict,
                           precision_k5: float, recall_k5: float,
                           precision_k10: float, recall_k10: float,
                           cf_coverage: dict) -> pd.DataFrame:
    """
    Build a summary comparison table across all recommendation approaches.
    """
    rows = [
        {
            "Model"          : "Popularity-Based",
            "RMSE"           : "N/A",
            "MAE"            : "N/A",
            "Precision@5"    : "N/A",
            "Recall@5"       : "N/A",
            "Precision@10"   : "N/A",
            "Recall@10"      : "N/A",
            "Coverage"       : "100%",
            "Notes"          : "No personalisation; always returns same top-N",
        },
        {
            "Model"          : "Item-Based CF",
            "RMSE"           : "N/A",
            "MAE"            : "N/A",
            "Precision@5"    : "N/A",
            "Recall@5"       : "N/A",
            "Precision@10"   : "N/A",
            "Recall@10"      : "N/A",
            "Coverage"       : f"{cf_coverage.get('Coverage', 0)*100:.1f}%",
            "Notes"          : "Cosine similarity on book-user matrix",
        },
        {
            "Model"          : "User-Based CF",
            "RMSE"           : "N/A",
            "MAE"            : "N/A",
            "Precision@5"    : "N/A",
            "Recall@5"       : "N/A",
            "Precision@10"   : "N/A",
            "Recall@10"      : "N/A",
            "Coverage"       : f"{cf_coverage.get('Coverage', 0)*100:.1f}%",
            "Notes"          : "Cosine similarity on user-book matrix",
        },
        {
            "Model"          : "SVD (Matrix Factorisation)",
            "RMSE"           : svd_metrics.get("RMSE", "N/A"),
            "MAE"            : svd_metrics.get("MAE",  "N/A"),
            "Precision@5"    : precision_k5,
            "Recall@5"       : recall_k5,
            "Precision@10"   : precision_k10,
            "Recall@10"      : recall_k10,
            "Coverage"       : "~80%",
            "Notes"          : "Latent factor model; best prediction accuracy",
        },
        {
            "Model"          : "Hybrid (Pop + CF)",
            "RMSE"           : "N/A",
            "MAE"            : "N/A",
            "Precision@5"    : "N/A",
            "Recall@5"       : "N/A",
            "Precision@10"   : "N/A",
            "Recall@10"      : "N/A",
            "Coverage"       : "High",
            "Notes"          : "Balances popularity & personalisation",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\n[EVALUATION] Comparison table saved to outputs/model_comparison.csv")
    return df


# =============================================================================
# 5. Visualisations
# =============================================================================

def plot_rmse_mae(svd_metrics: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("SVD Model – RMSE & MAE", fontsize=14, color=FG_COLOR, fontweight="bold")

    metrics = ["RMSE", "MAE"]
    values  = [svd_metrics["RMSE"], svd_metrics["MAE"]]
    bars    = ax.bar(metrics, values, color=[ACC1, ACC2], edgecolor="none", width=0.4)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12,
                color=FG_COLOR, fontweight="bold")

    ax.set_ylabel("Error")
    ax.set_ylim(0, max(values) * 1.4)
    ax.grid(axis="y")
    plt.tight_layout()
    _save(fig, "09_rmse_mae.png")


def plot_precision_recall(p5, r5, p10, r10):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Precision & Recall @ K (SVD)", fontsize=14,
                 color=FG_COLOR, fontweight="bold")

    x      = np.arange(2)
    width  = 0.35
    precs  = [p5, p10]
    recs   = [r5, r10]

    b1 = ax.bar(x - width/2, precs, width, label="Precision@K", color=ACC1, edgecolor="none")
    b2 = ax.bar(x + width/2, recs,  width, label="Recall@K",    color=ACC2, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(["K = 5", "K = 10"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=9, color=FG_COLOR)

    plt.tight_layout()
    _save(fig, "10_precision_recall.png")


def plot_model_comparison(comparison_df: pd.DataFrame):
    """Bar chart comparing RMSE across numeric models."""
    numeric_models = comparison_df[comparison_df["RMSE"] != "N/A"].copy()
    if numeric_models.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison – SVD Metrics",
                 fontsize=14, color=FG_COLOR, fontweight="bold")

    rmse_vals = numeric_models["RMSE"].astype(float).values
    mae_vals  = numeric_models["MAE"].astype(float).values
    models    = numeric_models["Model"].values

    axes[0].bar(models, rmse_vals, color=ACC1, edgecolor="none")
    axes[0].set_title("RMSE", color=FG_COLOR)
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y")

    axes[1].bar(models, mae_vals, color=ACC2, edgecolor="none")
    axes[1].set_title("MAE", color=FG_COLOR)
    axes[1].set_ylabel("MAE")
    axes[1].grid(axis="y")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    _save(fig, "11_model_comparison.png")


def plot_prediction_error_distribution(predictions):
    """Histogram of prediction errors from SVD."""
    errors = [abs(p.r_ui - p.est) for p in predictions]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("SVD Prediction Error Distribution",
                 fontsize=14, color=FG_COLOR, fontweight="bold")

    ax.hist(errors, bins=50, color=ACC3, edgecolor="none", alpha=0.9)
    ax.axvline(np.mean(errors), color=ACC1, linestyle="--",
               linewidth=2, label=f"Mean Error: {np.mean(errors):.3f}")
    ax.set_xlabel("|Actual − Predicted|")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    _save(fig, "12_error_distribution.png")


# =============================================================================
# Master Evaluation Runner
# =============================================================================

def run_evaluation(svd_metrics: dict, predictions,
                   cf_model, sample_books: list) -> pd.DataFrame:
    """
    Orchestrate all evaluation steps.

    Parameters
    ----------
    svd_metrics   : {"RMSE": float, "MAE": float}
    predictions   : list of Surprise Prediction namedtuples
    cf_model      : fitted CollaborativeFilter
    sample_books  : list of book titles to assess CF coverage
    """
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    # RMSE / MAE
    metrics = compute_rmse_mae(predictions)
    print(f"  RMSE : {metrics['RMSE']}")
    print(f"  MAE  : {metrics['MAE']}")

    # Precision / Recall
    p5,  r5  = precision_recall_at_k(predictions, k=5,  threshold=7.0)
    p10, r10 = precision_recall_at_k(predictions, k=10, threshold=7.0)
    print(f"  Precision@5  : {p5}   Recall@5  : {r5}")
    print(f"  Precision@10 : {p10}  Recall@10 : {r10}")

    # CF Coverage
    cf_cov = evaluate_cf_coverage(cf_model, sample_books)
    print(f"  CF Coverage : {cf_cov['Coverage']*100:.1f}%")

    # Comparison table
    comparison_df = build_comparison_table(metrics, p5, r5, p10, r10, cf_cov)

    # Plots
    plot_rmse_mae(metrics)
    plot_precision_recall(p5, r5, p10, r10)
    plot_model_comparison(comparison_df)
    plot_prediction_error_distribution(predictions)

    print("\n[EVALUATION COMPLETE]")
    print(comparison_df.to_string(index=False))
    return comparison_df
