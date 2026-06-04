"""
=============================================================================
preprocessing.py  –  Book Recommendation System
=============================================================================
MCA Capstone Project | Artificial Intelligence
Author  : Student
Purpose : Load, clean, and prepare the Books / Users / Ratings datasets
          for downstream EDA, modelling and evaluation.
=============================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ---------------------------------------------------------------------------
# 1. Raw Loaders
# ---------------------------------------------------------------------------

def load_raw_data(data_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three raw CSV files from `data_path` (or DATA_DIR if None).

    Returns
    -------
    books   : pd.DataFrame
    users   : pd.DataFrame
    ratings : pd.DataFrame
    """
    path = data_path or DATA_DIR

    print("=" * 60)
    print("  LOADING RAW DATASETS")
    print("=" * 60)

    books   = pd.read_csv(os.path.join(path, "Books.csv"),
                          encoding="latin-1", low_memory=False)
    users   = pd.read_csv(os.path.join(path, "Users.csv"),
                          encoding="latin-1", low_memory=False)
    ratings = pd.read_csv(os.path.join(path, "Ratings.csv"),
                          encoding="latin-1", low_memory=False)

    print(f"  Books   shape : {books.shape}")
    print(f"  Users   shape : {users.shape}")
    print(f"  Ratings shape : {ratings.shape}")
    print()
    return books, users, ratings


# ---------------------------------------------------------------------------
# 2. Column Standardisation
# ---------------------------------------------------------------------------

def standardise_columns(books: pd.DataFrame,
                        users: pd.DataFrame,
                        ratings: pd.DataFrame
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rename columns to snake_case for consistency throughout the project.
    """
    books_rename = {
        "ISBN"               : "isbn",
        "Book-Title"         : "book_title",
        "Book-Author"        : "book_author",
        "Year-Of-Publication": "year_of_publication",
        "Publisher"          : "publisher",
        "Image-URL-S"        : "image_url_s",
        "Image-URL-M"        : "image_url_m",
        "Image-URL-L"        : "image_url_l",
    }
    users_rename = {
        "User-ID" : "user_id",
        "Location": "location",
        "Age"     : "age",
    }
    ratings_rename = {
        "User-ID"    : "user_id",
        "ISBN"       : "isbn",
        "Book-Rating": "book_rating",
    }

    books   = books.rename(columns=books_rename)
    users   = users.rename(columns=users_rename)
    ratings = ratings.rename(columns=ratings_rename)

    print("[INFO] Columns standardised to snake_case.")
    return books, users, ratings


# ---------------------------------------------------------------------------
# 3. Clean Books
# ---------------------------------------------------------------------------

def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop duplicate ISBNs
    2. Strip leading/trailing whitespace from text fields
    3. Convert year_of_publication to numeric; replace invalid years with NaN
    4. Drop rows where both title AND author are missing
    5. Fill remaining NaN text columns with 'Unknown'
    """
    print("\n[CLEANING] Books dataset ...")

    before = len(books)
    books.drop_duplicates(subset=["isbn"], keep="first", inplace=True)
    print(f"  Duplicates removed : {before - len(books)}")

    # Convert year to numeric
    books["year_of_publication"] = pd.to_numeric(
        books["year_of_publication"], errors="coerce"
    )
    # Valid range: books published 1800 – 2024
    invalid_years = books["year_of_publication"].notna() & (
        (books["year_of_publication"] < 1800) |
        (books["year_of_publication"] > 2024)
    )
    books.loc[invalid_years, "year_of_publication"] = np.nan
    print(f"  Invalid years set to NaN : {invalid_years.sum()}")

    # Strip whitespace
    str_cols = ["book_title", "book_author", "publisher"]
    for col in str_cols:
        if col in books.columns:
            books[col] = books[col].astype(str).str.strip()

    # Drop rows missing both title and author
    before = len(books)
    books.dropna(subset=["book_title"], inplace=True)
    print(f"  Rows without title removed : {before - len(books)}")

    # Fill missing text cols
    for col in str_cols:
        books[col].replace("nan", "Unknown", inplace=True)
        books[col].fillna("Unknown", inplace=True)

    print(f"  Books after cleaning : {len(books)}")
    return books


# ---------------------------------------------------------------------------
# 4. Clean Users
# ---------------------------------------------------------------------------

def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """
    1. Remove duplicate user_ids
    2. Convert age to numeric; replace implausible ages (< 5 or > 100) with NaN
    3. Fill missing ages with median age
    4. Extract country from location string
    """
    print("\n[CLEANING] Users dataset ...")

    before = len(users)
    users.drop_duplicates(subset=["user_id"], keep="first", inplace=True)
    print(f"  Duplicates removed : {before - len(users)}")

    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    invalid_age = users["age"].notna() & ((users["age"] < 5) | (users["age"] > 100))
    users.loc[invalid_age, "age"] = np.nan
    print(f"  Invalid ages set to NaN : {invalid_age.sum()}")

    median_age = users["age"].median()
    users["age"].fillna(median_age, inplace=True)
    print(f"  Missing ages filled with median : {median_age:.1f}")

    # Extract country (last component of comma-separated location)
    users["country"] = (
        users["location"]
        .astype(str)
        .str.split(",")
        .str[-1]
        .str.strip()
        .str.title()
    )

    print(f"  Users after cleaning : {len(users)}")
    return users


# ---------------------------------------------------------------------------
# 5. Clean Ratings
# ---------------------------------------------------------------------------

def clean_ratings(ratings: pd.DataFrame,
                  books: pd.DataFrame,
                  users: pd.DataFrame) -> pd.DataFrame:
    """
    1. Remove duplicate (user_id, isbn) pairs
    2. Keep only ratings that reference existing books and users
    3. Separate explicit (1-10) and implicit (0) ratings
    """
    print("\n[CLEANING] Ratings dataset ...")

    before = len(ratings)
    ratings.drop_duplicates(subset=["user_id", "isbn"], keep="first", inplace=True)
    print(f"  Duplicates removed : {before - len(ratings)}")

    valid_isbns  = set(books["isbn"].unique())
    valid_users  = set(users["user_id"].unique())

    before = len(ratings)
    ratings = ratings[
        ratings["isbn"].isin(valid_isbns) &
        ratings["user_id"].isin(valid_users)
    ]
    print(f"  Orphan ratings removed : {before - len(ratings)}")
    print(f"  Ratings after cleaning : {len(ratings)}")
    return ratings


# ---------------------------------------------------------------------------
# 6. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(books: pd.DataFrame,
                      ratings: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build two derived datasets:

    explicit_ratings : only ratings where book_rating >= 1
    book_stats       : per-book aggregates (avg_rating, rating_count)
    """
    explicit_ratings = ratings[ratings["book_rating"] >= 1].copy()

    book_stats = (
        explicit_ratings
        .groupby("isbn")
        .agg(avg_rating=("book_rating", "mean"),
             rating_count=("book_rating", "count"))
        .reset_index()
    )

    books_enriched = books.merge(book_stats, on="isbn", how="left")
    books_enriched["avg_rating"]   = books_enriched["avg_rating"].fillna(0)
    books_enriched["rating_count"] = books_enriched["rating_count"].fillna(0).astype(int)

    print("\n[FEATURE ENG] book_stats created.")
    print(f"  explicit_ratings rows : {len(explicit_ratings)}")
    print(f"  books_enriched rows   : {len(books_enriched)}")

    return explicit_ratings, book_stats, books_enriched


# ---------------------------------------------------------------------------
# 7. Save Cleaned Data
# ---------------------------------------------------------------------------

def save_cleaned_data(books: pd.DataFrame,
                      users: pd.DataFrame,
                      ratings: pd.DataFrame,
                      explicit_ratings: pd.DataFrame,
                      books_enriched: pd.DataFrame) -> None:
    """Persist cleaned datasets to outputs/ directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    books.to_csv(os.path.join(OUTPUT_DIR, "books_clean.csv"), index=False)
    users.to_csv(os.path.join(OUTPUT_DIR, "users_clean.csv"), index=False)
    ratings.to_csv(os.path.join(OUTPUT_DIR, "ratings_clean.csv"), index=False)
    explicit_ratings.to_csv(os.path.join(OUTPUT_DIR, "explicit_ratings.csv"), index=False)
    books_enriched.to_csv(os.path.join(OUTPUT_DIR, "books_enriched.csv"), index=False)

    print("\n[SAVE] All cleaned datasets written to outputs/")


# ---------------------------------------------------------------------------
# 8. Master Pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(data_path: str = None):
    """
    Execute the full preprocessing pipeline and return all cleaned DataFrames.
    Called by main.py and the Jupyter notebook.
    """
    books, users, ratings = load_raw_data(data_path)
    books, users, ratings = standardise_columns(books, users, ratings)

    books   = clean_books(books)
    users   = clean_users(users)
    ratings = clean_ratings(ratings, books, users)

    explicit_ratings, book_stats, books_enriched = engineer_features(books, ratings)
    save_cleaned_data(books, users, ratings, explicit_ratings, books_enriched)

    print("\n[PREPROCESSING COMPLETE]")
    return books, users, ratings, explicit_ratings, book_stats, books_enriched


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing(data_path=os.path.join(BASE_DIR, "..", "dataset"))
