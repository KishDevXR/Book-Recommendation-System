"""
=============================================================================
recommender.py  –  Book Recommendation System
=============================================================================
MCA Capstone Project | Artificial Intelligence
Author  : Student
Purpose : Four recommendation engines:
          A) Popularity-Based
          B) Collaborative Filtering (User-Based & Item-Based, Cosine Sim)
          C) Matrix Factorisation (SVD via Surprise)
          D) Hybrid (Popularity + Collaborative)
=============================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# A. Popularity-Based Recommender
# =============================================================================

class PopularityRecommender:
    """
    Recommends books purely based on aggregated statistics:
        - Weighted score  = avg_rating * log(rating_count + 1)
        - Filters books with at least `min_ratings` reviews
    """

    def __init__(self, min_ratings: int = 50):
        self.min_ratings = min_ratings
        self.popular_books: pd.DataFrame = None

    def fit(self, books_enriched: pd.DataFrame) -> "PopularityRecommender":
        df = books_enriched[books_enriched["rating_count"] >= self.min_ratings].copy()
        df["weighted_score"] = (
            df["avg_rating"] * np.log1p(df["rating_count"])
        )
        self.popular_books = df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
        print(f"[PopularityRecommender] Fitted. Eligible books: {len(self.popular_books)}")
        return self

    def recommend(self, n: int = 10) -> pd.DataFrame:
        """Return top-n popular books."""
        if self.popular_books is None:
            raise RuntimeError("Call fit() first.")
        cols = ["book_title", "book_author", "year_of_publication",
                "publisher", "avg_rating", "rating_count", "weighted_score"]
        available = [c for c in cols if c in self.popular_books.columns]
        return self.popular_books[available].head(n).reset_index(drop=True)

    def top10(self)  -> pd.DataFrame: return self.recommend(10)
    def top20(self)  -> pd.DataFrame: return self.recommend(20)
    def top50(self)  -> pd.DataFrame: return self.recommend(50)

    def save(self):
        with open(os.path.join(MODEL_DIR, "popularity_model.pkl"), "wb") as f:
            pickle.dump(self, f)
        print("[PopularityRecommender] Model saved.")

    @staticmethod
    def load() -> "PopularityRecommender":
        with open(os.path.join(MODEL_DIR, "popularity_model.pkl"), "rb") as f:
            return pickle.load(f)


# =============================================================================
# B. Collaborative Filtering
# =============================================================================

class CollaborativeFilter:
    """
    Item-Based and User-Based Collaborative Filtering using Cosine Similarity.

    Only users who have rated at least `min_user_ratings` books and
    books that have at least `min_book_ratings` ratings are retained
    to keep the pivot matrix tractable.
    """

    def __init__(self, min_user_ratings: int = 50, min_book_ratings: int = 10):
        self.min_user_ratings = min_user_ratings
        self.min_book_ratings = min_book_ratings

        # Fitted artefacts
        self.pivot_table      : pd.DataFrame  = None   # books × users
        self.book_similarity  : np.ndarray    = None   # item-item cosine
        self.user_similarity  : np.ndarray    = None   # user-user cosine
        self.book_index       : pd.Index      = None
        self.user_index       : pd.Index      = None
        self.books_df         : pd.DataFrame  = None   # isbn → title etc.

    # ------------------------------------------------------------------
    def fit(self, explicit_ratings: pd.DataFrame,
            books: pd.DataFrame) -> "CollaborativeFilter":
        """Build pivot matrix and compute cosine similarities."""
        print("\n[CollaborativeFilter] Building pivot matrix …")

        self.books_df = books.set_index("isbn")

        # Filter active users
        user_counts = explicit_ratings["user_id"].value_counts()
        active_users = user_counts[user_counts >= self.min_user_ratings].index
        df = explicit_ratings[explicit_ratings["user_id"].isin(active_users)]

        # Filter popular books
        book_counts = df["isbn"].value_counts()
        popular_books = book_counts[book_counts >= self.min_book_ratings].index
        df = df[df["isbn"].isin(popular_books)]

        print(f"  Active users   : {df['user_id'].nunique()}")
        print(f"  Popular books  : {df['isbn'].nunique()}")

        # Pivot: rows = books, columns = users, values = ratings
        pivot = df.pivot_table(index="isbn", columns="user_id",
                               values="book_rating", fill_value=0)
        self.pivot_table = pivot
        self.book_index  = pivot.index
        self.user_index  = pivot.columns

        # Sparse matrix for cosine similarity
        sparse = csr_matrix(pivot.values)
        print("  Computing item-item cosine similarity …")
        self.book_similarity = cosine_similarity(sparse)
        print("  Computing user-user cosine similarity …")
        self.user_similarity = cosine_similarity(sparse.T)

        print("[CollaborativeFilter] Fit complete.")
        return self

    # ------------------------------------------------------------------
    def recommend_books(self, book_name: str, n: int = 10) -> pd.DataFrame:
        """
        Item-Based CF: given a book title, find n most similar books
        using item-item cosine similarity.
        """
        if self.pivot_table is None:
            raise RuntimeError("Call fit() first.")

        # Find ISBN matching the book name
        isbn = self._find_isbn(book_name)
        if isbn not in list(self.book_index):
            return pd.DataFrame({"Message": [f"Book '{book_name}' not in model pivot."]})

        idx = list(self.book_index).index(isbn)
        sim_scores = list(enumerate(self.book_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]   # exclude self

        recs = []
        for book_idx, score in sim_scores:
            rec_isbn = self.book_index[book_idx]
            if rec_isbn in self.books_df.index:
                row = self.books_df.loc[rec_isbn]
                recs.append({
                    "book_title"  : row.get("book_title",  rec_isbn),
                    "book_author" : row.get("book_author", "Unknown"),
                    "similarity"  : round(score, 4),
                })
        return pd.DataFrame(recs)

    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        User-Based CF: for a given user_id, find similar users and
        recommend books they rated highly that the target user hasn't read.
        """
        if self.pivot_table is None:
            raise RuntimeError("Call fit() first.")

        if user_id not in self.user_index:
            return pd.DataFrame({"Message": [f"User {user_id} not found in model."]})

        user_pos   = list(self.user_index).index(user_id)
        sim_scores = self.user_similarity[user_pos]
        # Top 20 most similar users
        similar_users = np.argsort(sim_scores)[::-1][1:21]

        # Books already rated by target user — look along the user column
        if user_id in self.pivot_table.columns:
            rated_by_user = set(self.pivot_table.index[self.pivot_table[user_id] > 0])
        else:
            rated_by_user = set()

        candidate_scores: dict = {}
        for su in similar_users:
            su_id = self.user_index[su]
            weight = sim_scores[su]
            # Books rated by similar user (col = user, row = book in pivot_table)
            for book_isbn in self.book_index:
                if book_isbn not in rated_by_user:
                    rating = self.pivot_table.loc[book_isbn, su_id] if su_id in self.pivot_table.columns else 0
                    if rating > 0:
                        candidate_scores[book_isbn] = candidate_scores.get(book_isbn, 0) + weight * rating

        if not candidate_scores:
            return pd.DataFrame({"Message": ["No recommendations found for this user."]})

        top_books = sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:n]
        recs = []
        for isbn in top_books:
            if isbn in self.books_df.index:
                row = self.books_df.loc[isbn]
                recs.append({
                    "book_title"  : row.get("book_title",  isbn),
                    "book_author" : row.get("book_author", "Unknown"),
                    "score"       : round(candidate_scores[isbn], 4),
                })
        return pd.DataFrame(recs)

    # ------------------------------------------------------------------
    def _find_isbn(self, book_name: str):
        """Fuzzy ISBN lookup by partial title match."""
        needle = book_name.lower().strip()
        for isbn in self.book_index:
            if isbn in self.books_df.index:
                title = str(self.books_df.loc[isbn].get("book_title", "")).lower()
                if needle in title or title in needle:
                    return isbn
        # Try broader search across all books_df
        for isbn, row in self.books_df.iterrows():
            title = str(row.get("book_title", "")).lower()
            if needle in title:
                return isbn
        return None

    # ------------------------------------------------------------------
    def save(self):
        with open(os.path.join(MODEL_DIR, "cf_model.pkl"), "wb") as f:
            pickle.dump(self, f)
        print("[CollaborativeFilter] Model saved.")

    @staticmethod
    def load() -> "CollaborativeFilter":
        with open(os.path.join(MODEL_DIR, "cf_model.pkl"), "rb") as f:
            return pickle.load(f)


# =============================================================================
# C. Matrix Factorisation – SVD (Surprise)
# =============================================================================

class SVDRecommender:
    """
    Uses the Surprise library's SVD algorithm for matrix factorisation.
    Trains on explicit (1-10) ratings.
    """

    def __init__(self, n_factors: int = 50, n_epochs: int = 20,
                 lr_all: float = 0.005, reg_all: float = 0.02):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr_all    = lr_all
        self.reg_all   = reg_all
        self.algo      = None
        self.trainset  = None

    def fit(self, explicit_ratings: pd.DataFrame,
            test_size: float = 0.2) -> dict:
        """
        Train SVD and return RMSE / MAE on the test split.
        """
        from surprise import Dataset, Reader, SVD
        from surprise.model_selection import train_test_split
        from surprise import accuracy

        print("\n[SVDRecommender] Training SVD …")

        # Sample to keep runtime reasonable (max 200K rows)
        df = explicit_ratings[["user_id", "isbn", "book_rating"]].copy()
        if len(df) > 200_000:
            df = df.sample(200_000, random_state=42)

        reader  = Reader(rating_scale=(1, 10))
        dataset = Dataset.load_from_df(df, reader)

        trainset, testset = train_test_split(dataset, test_size=test_size,
                                             random_state=42)
        self.trainset = trainset

        self.algo = SVD(n_factors=self.n_factors,
                        n_epochs=self.n_epochs,
                        lr_all=self.lr_all,
                        reg_all=self.reg_all,
                        verbose=False)
        self.algo.fit(trainset)

        predictions = self.algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae  = accuracy.mae(predictions,  verbose=False)

        metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}
        print(f"  SVD  RMSE : {rmse:.4f}")
        print(f"  SVD  MAE  : {mae:.4f}")
        return metrics, predictions

    def predict(self, user_id: int, isbn: str) -> float:
        """Predict the rating a user would give to a book."""
        if self.algo is None:
            raise RuntimeError("Call fit() first.")
        pred = self.algo.predict(str(user_id), str(isbn))
        return round(pred.est, 2)

    def save(self):
        with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "wb") as f:
            pickle.dump(self, f)
        print("[SVDRecommender] Model saved.")

    @staticmethod
    def load() -> "SVDRecommender":
        with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
            return pickle.load(f)


# =============================================================================
# D. Hybrid Recommender
# =============================================================================

class HybridRecommender:
    """
    Combines Popularity-Based and Collaborative Filtering scores:
        hybrid_score = alpha * popularity_score + (1-alpha) * cf_similarity
    """

    def __init__(self, popularity: PopularityRecommender,
                 cf: CollaborativeFilter,
                 alpha: float = 0.4):
        self.popularity = popularity
        self.cf         = cf
        self.alpha      = alpha

    def recommend_books(self, book_name: str, n: int = 10) -> pd.DataFrame:
        """
        For a given book, blend CF item-similarity with popularity score.
        """
        cf_recs = self.cf.recommend_books(book_name, n=n*2)
        if "Message" in cf_recs.columns:
            # Fall back to popularity
            return self.popularity.recommend(n)

        pop_df = self.popularity.popular_books.copy()
        max_ws = pop_df["weighted_score"].max() or 1.0
        pop_df["pop_norm"] = pop_df["weighted_score"] / max_ws
        pop_index = pop_df.set_index("book_title")["pop_norm"].to_dict()

        cf_recs["pop_score"] = cf_recs["book_title"].map(
            lambda t: pop_index.get(t, 0.0)
        )
        cf_recs["hybrid_score"] = (
            (1 - self.alpha) * cf_recs["similarity"] +
            self.alpha        * cf_recs["pop_score"]
        )
        result = (
            cf_recs
            .sort_values("hybrid_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        result.index = result.index + 1
        return result[["book_title", "book_author", "similarity",
                        "pop_score", "hybrid_score"]]

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        For a given user, blend user-based CF with popularity.
        """
        user_recs = self.cf.recommend_for_user(user_id, n=n*2)
        if "Message" in user_recs.columns:
            return self.popularity.recommend(n)

        pop_df    = self.popularity.popular_books.copy()
        max_ws    = pop_df["weighted_score"].max() or 1.0
        pop_df["pop_norm"] = pop_df["weighted_score"] / max_ws
        pop_index = pop_df.set_index("book_title")["pop_norm"].to_dict()

        user_recs["pop_score"] = user_recs["book_title"].map(
            lambda t: pop_index.get(t, 0.0)
        )
        # Normalise CF score
        max_score = user_recs["score"].max() or 1.0
        user_recs["cf_norm"] = user_recs["score"] / max_score

        user_recs["hybrid_score"] = (
            (1 - self.alpha) * user_recs["cf_norm"] +
            self.alpha        * user_recs["pop_score"]
        )
        result = (
            user_recs
            .sort_values("hybrid_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        result.index = result.index + 1
        return result[["book_title", "book_author", "score",
                        "pop_score", "hybrid_score"]]


# =============================================================================
# Public API Functions
# =============================================================================

def recommend_books(book_name: str, cf: CollaborativeFilter = None,
                    hybrid: HybridRecommender = None, n: int = 10) -> None:
    """
    Wrapper that prints formatted recommendations for a given book title.

    Uses Hybrid recommender if available, else falls back to CF, else raises.
    """
    engine = hybrid or cf
    if engine is None:
        raise ValueError("Provide at least a CollaborativeFilter instance.")

    print(f"\n{'═'*55}")
    print(f"  Book Recommendations for: '{book_name}'")
    print(f"{'═'*55}")

    result = engine.recommend_books(book_name, n=n)
    if "Message" in result.columns:
        print(f"  ⚠  {result.iloc[0, 0]}")
        return

    for i, row in result.iterrows():
        idx = i if isinstance(i, int) else i + 1
        title  = row.get("book_title",  "N/A")
        author = row.get("book_author", "N/A")
        print(f"  {idx:>2}. {title}")
        print(f"       Author : {author}")
    print()


def recommend_for_user(user_id: int, cf: CollaborativeFilter = None,
                       hybrid: HybridRecommender = None, n: int = 10) -> None:
    """
    Wrapper that prints formatted personalised recommendations for a user.
    """
    engine = hybrid or cf
    if engine is None:
        raise ValueError("Provide at least a CollaborativeFilter instance.")

    print(f"\n{'═'*55}")
    print(f"  Personalised Recommendations for User: {user_id}")
    print(f"{'═'*55}")

    result = engine.recommend_for_user(user_id, n=n)
    if "Message" in result.columns:
        print(f"  ⚠  {result.iloc[0, 0]}")
        return

    for i, row in result.iterrows():
        idx = i if isinstance(i, int) else i + 1
        title  = row.get("book_title",  "N/A")
        author = row.get("book_author", "N/A")
        print(f"  {idx:>2}. {title}")
        print(f"       Author : {author}")
    print()
