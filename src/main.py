# -*- coding: utf-8 -*-
"""
=============================================================================
main.py  -  Book Recommendation System
=============================================================================
MCA Capstone Project | Artificial Intelligence
Author  : Student
Purpose : End-to-end pipeline runner.
          Run this script to:
          1. Preprocess all datasets
          2. Run full EDA and save figures
          3. Train all recommendation models
          4. Evaluate models
          5. Demonstrate recommend_books() and recommend_for_user()
          6. Save all models to disk

Usage:
    cd Book-Recommendation-System
    python src/main.py
=============================================================================
"""

import os
import sys
import time
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure src/ is importable regardless of CWD
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

DATA_PATH = os.path.join(BASE_DIR, "..", "dataset")

from preprocessing import run_preprocessing
from eda           import run_eda
from recommender   import (PopularityRecommender, CollaborativeFilter,
                           SVDRecommender, HybridRecommender,
                           recommend_books, recommend_for_user)
from evaluation    import run_evaluation


def main():
    start = time.time()

    print("\n" + "=" * 60)
    print("   BOOK RECOMMENDATION SYSTEM  -  MCA CAPSTONE PROJECT")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------
    # STEP 1: Preprocessing
    # -----------------------------------------------------------------
    (books, users, ratings,
     explicit_ratings, book_stats,
     books_enriched) = run_preprocessing(data_path=DATA_PATH)

    # -----------------------------------------------------------------
    # STEP 2: EDA
    # -----------------------------------------------------------------
    run_eda(books, users, ratings, explicit_ratings, books_enriched)

    # -----------------------------------------------------------------
    # STEP 3A: Popularity-Based Recommender
    # -----------------------------------------------------------------
    print("\n" + "-"*55)
    print("  STEP 3A – Popularity-Based Recommender")
    print("-"*55)
    pop_model = PopularityRecommender(min_ratings=50)
    pop_model.fit(books_enriched)

    print("\n  ► Top 10 Most Popular Books:")
    print(pop_model.top10()[["book_title", "book_author",
                              "avg_rating", "rating_count"]].to_string(index=False))

    pop_model.save()

    # -----------------------------------------------------------------
    # STEP 3B: Collaborative Filtering
    # -----------------------------------------------------------------
    print("\n" + "-"*55)
    print("  STEP 3B – Collaborative Filtering")
    print("-"*55)
    cf_model = CollaborativeFilter(min_user_ratings=50, min_book_ratings=10)
    cf_model.fit(explicit_ratings, books)
    cf_model.save()

    # -----------------------------------------------------------------
    # STEP 3C: SVD (Matrix Factorisation)
    # -----------------------------------------------------------------
    print("\n" + "-"*55)
    print("  STEP 3C – SVD Matrix Factorisation")
    print("-"*55)
    svd_model = SVDRecommender(n_factors=50, n_epochs=20)
    svd_metrics, predictions = svd_model.fit(explicit_ratings)
    svd_model.save()

    # -----------------------------------------------------------------
    # STEP 3D: Hybrid Recommender
    # -----------------------------------------------------------------
    print("\n" + "-"*55)
    print("  STEP 3D – Hybrid Recommender")
    print("-"*55)
    hybrid = HybridRecommender(pop_model, cf_model, alpha=0.4)

    # -----------------------------------------------------------------
    # STEP 4: Evaluation
    # -----------------------------------------------------------------
    sample_titles = list(
        cf_model.books_df["book_title"].dropna().sample(
            min(20, len(cf_model.books_df)), random_state=42
        )
    ) if "book_title" in cf_model.books_df.columns else []

    comparison_df = run_evaluation(
        svd_metrics   = svd_metrics,
        predictions   = predictions,
        cf_model      = cf_model,
        sample_books  = sample_titles
    )

    # -----------------------------------------------------------------
    # STEP 5: Demo Recommendations
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  STEP 5 - RECOMMENDATION DEMOS")
    print("=" * 55)

    # Book-based recommendations
    demo_books = [
        "Harry Potter and the Sorcerer's Stone",
        "The Da Vinci Code",
        "Lord of the Rings",
    ]
    for book in demo_books:
        recommend_books(book, hybrid=hybrid)

    # User-based recommendations
    demo_users = list(cf_model.user_index[:3])
    for uid in demo_users:
        recommend_for_user(uid, hybrid=hybrid)

    # -----------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("                   PIPELINE COMPLETE")
    print(f"  Total time : {elapsed/60:.1f} minutes")
    print("=" * 60)
    print("\nOutputs saved to:")
    print(f"  Figures  -> {os.path.join(BASE_DIR, 'outputs', 'figures')}")
    print(f"  Models   -> {os.path.join(BASE_DIR, 'models')}")
    print(f"  CSVs     -> {os.path.join(BASE_DIR, 'outputs')}")


if __name__ == "__main__":
    main()
