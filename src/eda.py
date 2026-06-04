"""
=============================================================================
eda.py  –  Book Recommendation System
=============================================================================
MCA Capstone Project | Artificial Intelligence
Author  : Student
Purpose : Complete Exploratory Data Analysis with professional visualisations
          using Matplotlib and Seaborn.
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE  = "viridis"
BG_COLOR = "#0f0f1a"
FG_COLOR = "#e0e0f0"
ACC1     = "#7c3aed"          # violet accent
ACC2     = "#06b6d4"          # cyan accent
ACC3     = "#f59e0b"          # amber accent

plt.rcParams.update({
    "figure.facecolor"   : BG_COLOR,
    "axes.facecolor"     : "#1a1a2e",
    "axes.edgecolor"     : "#2a2a4a",
    "axes.labelcolor"    : FG_COLOR,
    "xtick.color"        : FG_COLOR,
    "ytick.color"        : FG_COLOR,
    "text.color"         : FG_COLOR,
    "grid.color"         : "#2a2a4a",
    "grid.linestyle"     : "--",
    "grid.alpha"         : 0.5,
    "font.family"        : "sans-serif",
    "axes.titlesize"     : 14,
    "axes.labelsize"     : 11,
    "legend.facecolor"   : "#1a1a2e",
    "legend.edgecolor"   : "#2a2a4a",
})

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [FIG] Saved → {name}")


# ---------------------------------------------------------------------------
# 1. Dataset Overview
# ---------------------------------------------------------------------------

def eda_overview(books, users, ratings, explicit_ratings):
    """Print shape, dtypes, missing values, duplicate counts."""
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS – OVERVIEW")
    print("=" * 60)

    for name, df in [("Books", books), ("Users", users),
                     ("Ratings", ratings), ("Explicit Ratings", explicit_ratings)]:
        print(f"\n{'─'*50}")
        print(f"  {name}  {df.shape}")
        print(f"{'─'*50}")
        mv = df.isnull().sum()
        mv = mv[mv > 0]
        if len(mv):
            print("  Missing values:")
            print(mv.to_string())
        else:
            print("  No missing values.")
        dups = df.duplicated().sum()
        print(f"  Duplicate rows : {dups}")


# ---------------------------------------------------------------------------
# 2. Rating Distribution
# ---------------------------------------------------------------------------

def plot_rating_distribution(ratings, explicit_ratings):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Rating Distribution", fontsize=16, color=FG_COLOR, fontweight="bold")

    # All ratings (including 0 = implicit)
    ax = axes[0]
    cnt = ratings["book_rating"].value_counts().sort_index()
    bars = ax.bar(cnt.index, cnt.values, color=[ACC1 if i == 0 else ACC2 for i in cnt.index],
                  edgecolor="none", width=0.8)
    ax.set_title("All Ratings (0 = Implicit)", color=FG_COLOR)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.grid(axis="y")
    for bar, val in zip(bars, cnt.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val/1000:.0f}K", ha="center", va="bottom", fontsize=8, color=FG_COLOR)

    # Explicit ratings only
    ax = axes[1]
    cnt2 = explicit_ratings["book_rating"].value_counts().sort_index()
    bars2 = ax.bar(cnt2.index, cnt2.values, color=ACC1, edgecolor="none", width=0.8)
    ax.set_title("Explicit Ratings (1–10)", color=FG_COLOR)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.grid(axis="y")
    for bar, val in zip(bars2, cnt2.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val/1000:.0f}K", ha="center", va="bottom", fontsize=8, color=FG_COLOR)

    plt.tight_layout()
    _save(fig, "01_rating_distribution.png")


# ---------------------------------------------------------------------------
# 3. Top Rated Books
# ---------------------------------------------------------------------------

def plot_top_rated_books(books_enriched, n=10):
    # Weighted score: need at least 50 ratings
    eligible = books_enriched[books_enriched["rating_count"] >= 50].copy()
    eligible = eligible.nlargest(n, "avg_rating")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Top {n} Highest Rated Books (≥50 ratings)", fontsize=16,
                 color=FG_COLOR, fontweight="bold")

    labels = [t[:40] + "…" if len(t) > 40 else t for t in eligible["book_title"]]
    bars   = ax.barh(range(len(labels)), eligible["avg_rating"],
                     color=sns.color_palette("viridis", n), edgecolor="none")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Average Rating")
    ax.set_xlim(0, 11)
    ax.invert_yaxis()
    ax.grid(axis="x")
    for bar, cnt in zip(bars, eligible["rating_count"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.2f}  ({cnt} ratings)",
                va="center", fontsize=8, color=FG_COLOR)

    plt.tight_layout()
    _save(fig, "02_top_rated_books.png")


# ---------------------------------------------------------------------------
# 4. Most Active Users
# ---------------------------------------------------------------------------

def plot_most_active_users(explicit_ratings, n=15):
    top_users = (
        explicit_ratings["user_id"]
        .value_counts()
        .head(n)
        .reset_index()
    )
    top_users.columns = ["user_id", "rating_count"]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Top {n} Most Active Users", fontsize=16, color=FG_COLOR, fontweight="bold")

    palette = sns.color_palette("plasma", n)
    bars    = ax.bar(top_users["user_id"].astype(str), top_users["rating_count"],
                     color=palette, edgecolor="none")
    ax.set_xlabel("User ID")
    ax.set_ylabel("Number of Ratings")
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8, color=FG_COLOR)

    plt.tight_layout()
    _save(fig, "03_most_active_users.png")


# ---------------------------------------------------------------------------
# 5. Publication Year Analysis
# ---------------------------------------------------------------------------

def plot_publication_year(books):
    year_data = books["year_of_publication"].dropna()
    year_data = year_data[(year_data >= 1900) & (year_data <= 2005)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Publication Year Analysis", fontsize=16, color=FG_COLOR, fontweight="bold")

    # Histogram
    ax = axes[0]
    ax.hist(year_data, bins=50, color=ACC1, edgecolor="none", alpha=0.9)
    ax.set_title("Distribution of Publication Years", color=FG_COLOR)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Books")
    ax.grid(axis="y")

    # Books per decade
    ax = axes[1]
    decade = ((year_data // 10) * 10).value_counts().sort_index()
    ax.bar(decade.index.astype(str), decade.values, color=ACC2, edgecolor="none")
    ax.set_title("Books Published per Decade", color=FG_COLOR)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y")

    plt.tight_layout()
    _save(fig, "04_publication_year.png")


# ---------------------------------------------------------------------------
# 6. Age Distribution
# ---------------------------------------------------------------------------

def plot_age_distribution(users):
    age_data = users["age"].dropna()
    age_data = age_data[(age_data >= 5) & (age_data <= 100)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("User Age Distribution", fontsize=16, color=FG_COLOR, fontweight="bold")

    ax = axes[0]
    ax.hist(age_data, bins=40, color=ACC3, edgecolor="none", alpha=0.9)
    ax.axvline(age_data.mean(), color=ACC1, linewidth=2, linestyle="--",
               label=f"Mean: {age_data.mean():.1f}")
    ax.axvline(age_data.median(), color=ACC2, linewidth=2, linestyle="--",
               label=f"Median: {age_data.median():.1f}")
    ax.set_title("Age Histogram", color=FG_COLOR)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y")

    # Box plot by age group
    ax = axes[1]
    bins   = [0, 18, 25, 35, 50, 65, 100]
    labels = ["<18", "18-25", "26-35", "36-50", "51-65", "65+"]
    users  = users.copy()
    users["age_group"] = pd.cut(users["age"], bins=bins, labels=labels)
    grp_cnt = users["age_group"].value_counts().reindex(labels)
    ax.bar(grp_cnt.index, grp_cnt.values,
           color=sns.color_palette("magma", len(labels)), edgecolor="none")
    ax.set_title("Users per Age Group", color=FG_COLOR)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    ax.grid(axis="y")

    plt.tight_layout()
    _save(fig, "05_age_distribution.png")


# ---------------------------------------------------------------------------
# 7. Top Authors & Publishers
# ---------------------------------------------------------------------------

def plot_top_authors_publishers(books, n=10):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Top Authors & Publishers", fontsize=16, color=FG_COLOR, fontweight="bold")

    top_authors = books["book_author"].value_counts().head(n)
    ax = axes[0]
    ax.barh(top_authors.index[::-1], top_authors.values[::-1],
            color=sns.color_palette("cool", n)[::-1], edgecolor="none")
    ax.set_title(f"Top {n} Authors by Book Count", color=FG_COLOR)
    ax.set_xlabel("Number of Books")
    ax.grid(axis="x")

    top_pub = books["publisher"].value_counts().head(n)
    ax = axes[1]
    short = [p[:30] + "…" if len(p) > 30 else p for p in top_pub.index]
    ax.barh(range(n), top_pub.values[::-1],
            color=sns.color_palette("YlOrRd", n)[::-1], edgecolor="none")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short[::-1])
    ax.set_title(f"Top {n} Publishers by Book Count", color=FG_COLOR)
    ax.set_xlabel("Number of Books")
    ax.grid(axis="x")

    plt.tight_layout()
    _save(fig, "06_top_authors_publishers.png")


# ---------------------------------------------------------------------------
# 8. Ratings per Book (Long-Tail)
# ---------------------------------------------------------------------------

def plot_ratings_long_tail(books_enriched):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Ratings Distribution (Long-Tail Effect)",
                 fontsize=16, color=FG_COLOR, fontweight="bold")

    rc = books_enriched["rating_count"]
    ax.hist(rc[rc > 0], bins=100, color=ACC1, edgecolor="none", log=True)
    ax.set_xlabel("Number of Ratings per Book")
    ax.set_ylabel("Number of Books (log scale)")
    ax.axvline(rc.mean(), color=ACC2, linestyle="--", linewidth=1.5,
               label=f"Mean: {rc.mean():.1f}")
    ax.axvline(rc.median(), color=ACC3, linestyle="--", linewidth=1.5,
               label=f"Median: {rc.median():.0f}")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    _save(fig, "07_long_tail.png")


# ---------------------------------------------------------------------------
# 9. Heatmap – Correlation
# ---------------------------------------------------------------------------

def plot_correlation(books_enriched, users):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Correlation Analysis", fontsize=16, color=FG_COLOR, fontweight="bold")

    # Books numeric correlation
    num_books = books_enriched[["year_of_publication", "avg_rating", "rating_count"]].dropna()
    corr      = num_books.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis",
                linewidths=0.5, linecolor="#0f0f1a", ax=axes[0],
                annot_kws={"size": 12, "color": "white"})
    axes[0].set_title("Books Numeric Correlation", color=FG_COLOR)
    axes[0].tick_params(colors=FG_COLOR)

    # Users numeric
    num_users = users[["age"]].dropna()
    num_users["user_id"] = users.loc[num_users.index, "user_id"]
    corr2 = users[["user_id", "age"]].corr()
    sns.heatmap(corr2, annot=True, fmt=".2f", cmap="magma",
                linewidths=0.5, linecolor="#0f0f1a", ax=axes[1],
                annot_kws={"size": 12, "color": "white"})
    axes[1].set_title("Users Correlation", color=FG_COLOR)
    axes[1].tick_params(colors=FG_COLOR)

    plt.tight_layout()
    _save(fig, "08_correlation.png")


# ---------------------------------------------------------------------------
# 10. Summary Dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(books, users, ratings, explicit_ratings, books_enriched):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG_COLOR)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle("Book Recommendation System – EDA Dashboard",
                 fontsize=20, color=FG_COLOR, fontweight="bold", y=1.01)

    # ── KPI boxes (row 0, col 0)
    ax_kpi = fig.add_subplot(gs[0, 0])
    ax_kpi.set_facecolor("#1a1a2e")
    ax_kpi.axis("off")
    kpis = [
        ("Total Books",   f"{len(books):,}"),
        ("Total Users",   f"{len(users):,}"),
        ("Total Ratings", f"{len(ratings):,}"),
        ("Explicit Rtg.", f"{len(explicit_ratings):,}"),
        ("Avg Rating",    f"{explicit_ratings['book_rating'].mean():.2f}"),
        ("Unique Authors",f"{books['book_author'].nunique():,}"),
    ]
    for i, (lbl, val) in enumerate(kpis):
        y_pos = 0.92 - i * 0.16
        ax_kpi.text(0.05, y_pos, lbl, transform=ax_kpi.transAxes,
                    fontsize=10, color="#9ca3af")
        ax_kpi.text(0.6, y_pos, val, transform=ax_kpi.transAxes,
                    fontsize=12, color=ACC2, fontweight="bold")
    ax_kpi.set_title("Key Metrics", color=FG_COLOR, pad=10)

    # ── Rating distribution bar (row 0, col 1-2)
    ax_rat = fig.add_subplot(gs[0, 1:])
    cnt = explicit_ratings["book_rating"].value_counts().sort_index()
    ax_rat.bar(cnt.index, cnt.values, color=sns.color_palette("viridis", 10), edgecolor="none")
    ax_rat.set_title("Explicit Rating Distribution (1–10)", color=FG_COLOR)
    ax_rat.set_xlabel("Rating")
    ax_rat.set_ylabel("Count")
    ax_rat.grid(axis="y")

    # ── Top 5 Authors (row 1, col 0)
    ax_auth = fig.add_subplot(gs[1, 0])
    top5 = books["book_author"].value_counts().head(5)
    ax_auth.barh(top5.index[::-1], top5.values[::-1],
                 color=sns.color_palette("plasma", 5), edgecolor="none")
    ax_auth.set_title("Top 5 Authors", color=FG_COLOR)
    ax_auth.set_xlabel("Books")
    ax_auth.grid(axis="x")

    # ── Publication decade (row 1, col 1)
    ax_yr = fig.add_subplot(gs[1, 1])
    year_data = books["year_of_publication"].dropna()
    year_data = year_data[(year_data >= 1900) & (year_data <= 2005)]
    decade = ((year_data // 10) * 10).value_counts().sort_index()
    ax_yr.bar(decade.index.astype(str), decade.values, color=ACC1, edgecolor="none")
    ax_yr.set_title("Books per Decade", color=FG_COLOR)
    ax_yr.set_xlabel("Decade")
    ax_yr.tick_params(axis="x", rotation=45)
    ax_yr.grid(axis="y")

    # ── Age histogram (row 1, col 2)
    ax_age = fig.add_subplot(gs[1, 2])
    age_data = users["age"].dropna()
    age_data = age_data[(age_data >= 5) & (age_data <= 100)]
    ax_age.hist(age_data, bins=30, color=ACC3, edgecolor="none", alpha=0.9)
    ax_age.set_title("User Age Distribution", color=FG_COLOR)
    ax_age.set_xlabel("Age")
    ax_age.set_ylabel("Count")
    ax_age.grid(axis="y")

    _save(fig, "00_summary_dashboard.png")


# ---------------------------------------------------------------------------
# Master EDA Runner
# ---------------------------------------------------------------------------

def run_eda(books, users, ratings, explicit_ratings, books_enriched):
    """Run all EDA steps and save figures."""
    print("\n" + "=" * 60)
    print("  RUNNING FULL EDA")
    print("=" * 60)

    eda_overview(books, users, ratings, explicit_ratings)
    plot_rating_distribution(ratings, explicit_ratings)
    plot_top_rated_books(books_enriched)
    plot_most_active_users(explicit_ratings)
    plot_publication_year(books)
    plot_age_distribution(users)
    plot_top_authors_publishers(books)
    plot_ratings_long_tail(books_enriched)
    plot_correlation(books_enriched, users)
    plot_summary_dashboard(books, users, ratings, explicit_ratings, books_enriched)

    print("\n[EDA COMPLETE] All figures saved to outputs/figures/")
