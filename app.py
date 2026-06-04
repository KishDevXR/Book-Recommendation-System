"""
=============================================================================
app.py  –  Book Recommendation System  |  Streamlit Frontend
=============================================================================
MCA Capstone Project | Artificial Intelligence
Author  : Student
Purpose : Interactive web UI for the Book Recommendation System.

Features:
  1. Top Rated Books Dashboard
  2. Search & Explore any book
  3. Get Similar Books (CF / Hybrid)
  4. Personalised Recommendations for a User

Run:
  cd Book-Recommendation-System
  streamlit run app.py
=============================================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR   = os.path.join(BASE_DIR, "..", "dataset")

# ── Streamlit page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title = "📚 Book Recommendation System",
    page_icon  = "📚",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Light elegant background ── */
.stApp {
    background: #f4f9fc;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0f2fe;
}

/* ── Cards ── */
.brs-card {
    background: #ffffff;
    border: 1px solid #bae6fd;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 10px 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.1), 0 2px 4px -1px rgba(14, 165, 233, 0.06);
}
.brs-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.2), 0 4px 6px -2px rgba(14, 165, 233, 0.1);
}

/* ── Metric card ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #7dd3fc;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(14, 165, 233, 0.05);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #0284c7, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 4px;
    font-weight: 500;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #0369a1, #0284c7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
}

/* ── Book rec card ── */
.rec-book {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
}
.rec-rank {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0284c7;
    margin-right: 8px;
}
.rec-title { font-weight: 600; color: #0f172a; }
.rec-author { color: #475569; font-size: 0.8rem; }

/* ── Pill badge ── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.pill-purple { background: rgba(2, 132, 199, 0.1); color: #0284c7; border: 1px solid #7dd3fc; }
.pill-cyan   { background: rgba(14, 165, 233, 0.1); color: #0ea5e9; border: 1px solid #bae6fd; }
.pill-amber  { background: rgba(3, 105, 161, 0.1); color: #0369a1; border: 1px solid #e0f2fe; }

/* ── Divider ── */
hr { border-color: #e0f2fe; }

/* ── Hide Streamlit menu ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    """Load cleaned data or fall back to raw CSVs."""
    # Try cleaned first
    books_path   = os.path.join(OUTPUT_DIR, "books_enriched.csv")
    users_path   = os.path.join(OUTPUT_DIR, "users_clean.csv")
    ratings_path = os.path.join(OUTPUT_DIR, "explicit_ratings.csv")

    if all(os.path.exists(p) for p in [books_path, users_path, ratings_path]):
        books   = pd.read_csv(books_path)
        users   = pd.read_csv(users_path)
        ratings = pd.read_csv(ratings_path)
    else:
        # Load raw and do minimal prep
        books   = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"),
                              encoding="latin-1", low_memory=False)
        users   = pd.read_csv(os.path.join(DATA_DIR, "Users.csv"),   encoding="latin-1")
        ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), encoding="latin-1")
        books.columns   = [c.lower().replace("-","_").replace(" ","_") for c in books.columns]
        users.columns   = [c.lower().replace("-","_").replace(" ","_") for c in users.columns]
        ratings.columns = [c.lower().replace("-","_").replace(" ","_") for c in ratings.columns]
        books.rename(columns={"bookx2dtitle":"book_title","bookx2dauthor":"book_author",
                               "yearx2dofx2dpublication":"year_of_publication"}, inplace=True)
        ratings = ratings[ratings.get("book_rating", ratings.get("bookx2drating", pd.Series(dtype=int))) >= 1]
    return books, users, ratings


@st.cache_resource(show_spinner=False)
def load_models():
    """Load pre-trained models if available."""
    models = {}
    for name, filename in [("popularity", "popularity_model.pkl"),
                            ("cf",         "cf_model.pkl"),
                            ("svd",        "svd_model.pkl")]:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def _star_rating(score: float, max_score: float = 10.0) -> str:
    stars = int(round((score / max_score) * 5))
    return "★" * stars + "☆" * (5 - stars)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:3rem;'>📚</div>
        <div style='font-size:1.3rem; font-weight:800; color:#0f172a;'>Book Recommender</div>
        <div style='font-size:0.75rem; color:#475569;'>MCA AI Capstone Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Search & Explore",
         "📖 Similar Books", "👤 User Recommendations",
         "📊 Model Evaluation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; text-align:center;'>
        Dataset: Book-Crossing<br>
        Algorithms: CF · SVD · Hybrid<br>
        Built with Streamlit 🎈
    </div>
    """, unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading datasets …"):
    books, users, ratings = load_data()

with st.spinner("Loading models …"):
    models = load_models()

pop_model = models.get("popularity")
cf_model  = models.get("cf")


# =============================================================================
# PAGE 1 – Dashboard
# =============================================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="section-header">📚 Book Recommendation System</div>', unsafe_allow_html=True)
    st.markdown("*MCA Capstone Project · AI & Machine Learning · Book-Crossing Dataset*")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("Total Books",    f"{len(books):,}"),
        ("Total Users",    f"{len(users):,}"),
        ("Total Ratings",  f"{len(ratings):,}"),
        ("Avg Rating",     f"{ratings['book_rating'].mean():.2f}" if 'book_rating' in ratings.columns else "N/A"),
        ("Unique Authors", f"{books['book_author'].nunique():,}" if 'book_author' in books.columns else "N/A"),
    ]
    for col, (label, value) in zip([col1,col2,col3,col4,col5], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Rating Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        if 'book_rating' in ratings.columns:
            cnt = ratings["book_rating"].value_counts().sort_index()
            palette = sns.color_palette("Blues_d", len(cnt))
            ax.bar(cnt.index, cnt.values, color=palette, edgecolor="none")
            ax.set_xlabel("Rating", color="#475569")
            ax.set_ylabel("Count", color="#475569")
            ax.tick_params(colors="#475569")
            ax.grid(axis="y", color="#e2e8f0", linestyle="--", alpha=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        st.markdown("**Books Published per Decade**")
        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        if 'year_of_publication' in books.columns:
            yr = pd.to_numeric(books["year_of_publication"], errors="coerce")
            yr = yr[(yr >= 1900) & (yr <= 2005)]
            decade = ((yr // 10) * 10).value_counts().sort_index()
            ax.bar(decade.index.astype(str), decade.values, color="#0ea5e9", edgecolor="none")
            ax.tick_params(axis="x", rotation=45, colors="#475569")
            ax.tick_params(axis="y", colors="#475569")
            ax.grid(axis="y", color="#e2e8f0", linestyle="--", alpha=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Top books table
    st.markdown("---")
    st.markdown("**🏆 Top 10 Popular Books**")

    if pop_model is not None:
        top10 = pop_model.top10()
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            title  = row.get("book_title",  "N/A")
            author = row.get("book_author", "N/A")
            avg    = row.get("avg_rating",  0)
            cnt    = row.get("rating_count", 0)
            st.markdown(f"""
            <div class="brs-card">
                <span class="rec-rank">#{i}</span>
                <span class="rec-title">{title}</span><br>
                <span class="rec-author">✍  {author}</span>&nbsp;&nbsp;
                <span class="pill pill-purple">⭐ {avg:.2f}</span>
                <span class="pill pill-cyan">📝 {int(cnt)} ratings</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Compute on-the-fly
        if all(c in books.columns for c in ["book_title","avg_rating","rating_count"]):
            df = books[books.get("rating_count", pd.Series([0]*len(books))) >= 50].copy()
            if len(df) == 0:
                df = books.copy()
            df = df.nlargest(10, "avg_rating") if "avg_rating" in df.columns else df.head(10)
            for i, (_, row) in enumerate(df.iterrows(), 1):
                st.markdown(f"**{i}.** {row.get('book_title','N/A')} — *{row.get('book_author','N/A')}*")
        else:
            st.info("Run the pipeline first (src/main.py) to train the popularity model.")


# =============================================================================
# PAGE 2 – Search & Explore
# =============================================================================
elif page == "🔍 Search & Explore":
    st.markdown('<div class="section-header">🔍 Search & Explore Books</div>', unsafe_allow_html=True)

    query = st.text_input("Search by book title or author …",
                          placeholder="e.g. Harry Potter, J.K. Rowling …")

    if query:
        mask = (
            books["book_title"].str.contains(query, case=False, na=False)  |
            books["book_author"].str.contains(query, case=False, na=False)
        ) if "book_title" in books.columns else pd.Series([False]*len(books))

        results = books[mask].head(20)

        if len(results) == 0:
            st.warning(f"No books found matching '{query}'.")
        else:
            st.success(f"Found **{len(results)}** book(s) matching '{query}'")
            for _, row in results.iterrows():
                title  = row.get("book_title",          "N/A")
                author = row.get("book_author",          "N/A")
                year   = int(row.get("year_of_publication", 0) or 0)
                pub    = row.get("publisher",            "N/A")
                avg    = row.get("avg_rating",            0)
                cnt    = row.get("rating_count",          0)

                with st.expander(f"📖 {title}"):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**Author:** {author}")
                        st.markdown(f"**Publisher:** {pub}")
                        st.markdown(f"**Year:** {year if year > 0 else 'Unknown'}")
                    with col_b:
                        if avg and avg > 0:
                            st.markdown(f"**Avg Rating:** {avg:.2f} / 10")
                            st.markdown(f"**# Ratings:** {int(cnt)}")
                        else:
                            st.markdown("*No rating data*")

    else:
        st.info("💡 Type any book title or author name to search the dataset.")

        # Show top 5 random popular books as inspiration
        if pop_model is not None:
            st.markdown("---")
            st.markdown("**✨ Popular Right Now**")
            sample = pop_model.popular_books.sample(5, random_state=42)
            for _, row in sample.iterrows():
                st.markdown(f"""
                <div class="brs-card">
                    <span class="rec-title">{row.get('book_title','N/A')}</span><br>
                    <span class="rec-author">✍  {row.get('book_author','N/A')}</span>
                    <span class="pill pill-amber" style="float:right">⭐ {row.get('avg_rating',0):.2f}</span>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3 – Similar Books
# =============================================================================
elif page == "📖 Similar Books":
    st.markdown('<div class="section-header">📖 Find Similar Books</div>', unsafe_allow_html=True)

    if cf_model is None:
        st.error("Collaborative Filtering model not found. Please run **src/main.py** first.")
        st.stop()

    # Book selector
    available_titles = []
    if cf_model.books_df is not None and "book_title" in cf_model.books_df.columns:
        available_titles = sorted(cf_model.books_df["book_title"].dropna().unique().tolist())

    if available_titles:
        book_choice = st.selectbox(
            "Choose a book to find similar titles:",
            options=available_titles,
            index=0,
        )
    else:
        book_choice = st.text_input("Enter a book title:", placeholder="Harry Potter …")

    n_recs = st.slider("Number of recommendations:", min_value=3, max_value=20, value=10)
    rec_type = st.radio("Recommendation Engine:", ["Hybrid (Popularity + CF)", "Item-Based CF Only"],
                        horizontal=True)

    if st.button("🔍 Find Similar Books", type="primary"):
        with st.spinner("Finding similar books …"):
            if rec_type == "Hybrid (Popularity + CF)" and pop_model is not None:
                from recommender import HybridRecommender
                hybrid = HybridRecommender(pop_model, cf_model, alpha=0.4)
                recs   = hybrid.recommend_books(book_choice, n=n_recs)
            else:
                recs = cf_model.recommend_books(book_choice, n=n_recs)

        if "Message" in recs.columns:
            st.warning(recs.iloc[0, 0])
        elif len(recs) == 0:
            st.warning("No similar books found.")
        else:
            st.success(f"**Books similar to '{book_choice}'**")
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                title  = row.get("book_title",  "N/A")
                author = row.get("book_author", "N/A")
                sim    = row.get("similarity",  row.get("hybrid_score", 0))
                st.markdown(f"""
                <div class="brs-card">
                    <span class="rec-rank">#{i}</span>
                    <span class="rec-title">{title}</span><br>
                    <span class="rec-author">✍  {author}</span>
                    <span class="pill pill-purple" style="float:right">
                        Similarity: {float(sim):.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4 – User Recommendations
# =============================================================================
elif page == "👤 User Recommendations":
    st.markdown('<div class="section-header">👤 Personalised Recommendations</div>', unsafe_allow_html=True)

    if cf_model is None:
        st.error("Collaborative Filtering model not found. Please run **src/main.py** first.")
        st.stop()

    available_users = sorted(cf_model.user_index.tolist()) if cf_model.user_index is not None else []

    if available_users:
        user_id = st.selectbox("Select a User ID:", options=available_users)
    else:
        user_id = st.number_input("Enter User ID:", min_value=1, value=276726)

    n_recs = st.slider("Number of recommendations:", min_value=3, max_value=20, value=10)

    if st.button("🎯 Get My Recommendations", type="primary"):
        with st.spinner("Generating personalised recommendations …"):
            if pop_model is not None:
                from recommender import HybridRecommender
                hybrid = HybridRecommender(pop_model, cf_model, alpha=0.4)
                recs   = hybrid.recommend_for_user(int(user_id), n=n_recs)
            else:
                recs = cf_model.recommend_for_user(int(user_id), n=n_recs)

        if "Message" in recs.columns:
            st.warning(recs.iloc[0, 0])
        elif len(recs) == 0:
            st.warning("No recommendations generated.")
        else:
            st.success(f"**Top {n_recs} Recommendations for User {user_id}**")
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                title  = row.get("book_title",  "N/A")
                author = row.get("book_author", "N/A")
                score  = row.get("hybrid_score", row.get("score", 0))
                st.markdown(f"""
                <div class="brs-card">
                    <span class="rec-rank">#{i}</span>
                    <span class="rec-title">{title}</span><br>
                    <span class="rec-author">✍  {author}</span>
                    <span class="pill pill-cyan" style="float:right">
                        Score: {float(score):.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5 – Model Evaluation
# =============================================================================
elif page == "📊 Model Evaluation":
    st.markdown('<div class="section-header">📊 Model Evaluation & Comparison</div>', unsafe_allow_html=True)

    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path)
        st.markdown("**Model Comparison Table**")
        st.dataframe(df.style.set_properties(**{
            "background-color": "#ffffff",
            "color": "#334155",
            "border-color": "#e2e8f0"
        }), use_container_width=True)

        # Highlight numeric metrics from SVD row
        numeric_models = df[df["RMSE"].apply(
            lambda x: str(x).replace('.','',1).isdigit()
        )].copy()
        if len(numeric_models) > 0:
            st.markdown("---")
            st.markdown("**SVD Model Metrics**")
            c1, c2, c3, c4 = st.columns(4)
            row = numeric_models.iloc[0]
            for col, key in zip([c1, c2, c3, c4],
                                ["RMSE", "MAE", "Precision@10", "Recall@10"]):
                val = row.get(key, "N/A")
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{key}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Evaluation results not found. Run **src/main.py** to train and evaluate models.")

    # Show saved figures
    fig_dir = os.path.join(BASE_DIR, "outputs", "figures")
    if os.path.isdir(fig_dir):
        figs = sorted([f for f in os.listdir(fig_dir) if f.endswith(".png")])
        if figs:
            st.markdown("---")
            st.markdown("**📈 Generated Visualisations**")
            for fig_name in figs:
                fig_path = os.path.join(fig_dir, fig_name)
                label = fig_name.replace("_", " ").replace(".png", "").title()
                with st.expander(f"📊 {label}"):
                    st.image(fig_path, use_column_width=True)
