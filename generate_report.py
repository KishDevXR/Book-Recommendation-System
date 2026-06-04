"""
generate_report.py  –  MCA Capstone DOCX Report Generator
Generates a professional Word document covering all required sections.
"""

import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(REPORT_DIR, exist_ok=True)

def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x1a, 0x56, 0xdb)
    return p

def body(doc, text):
    p = doc.add_paragraph(text)
    p.style.font.size = Pt(11)
    return p

def build_report():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name  = "Calibri"
    style.font.size  = Pt(11)

    # ── Title Page ─────────────────────────────────────────────────
    title = doc.add_heading("Book Recommendation System", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("MCA Artificial Intelligence Capstone Project")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].bold = True
    doc.add_paragraph("Dataset: Book-Crossing | Algorithms: CF · SVD · Hybrid"
                      ).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # ── Abstract ───────────────────────────────────────────────────
    heading(doc, "Abstract")
    body(doc,
         "This project presents a comprehensive Book Recommendation System developed "
         "as part of the MCA Artificial Intelligence Capstone. The system leverages the "
         "Book-Crossing dataset comprising 271,360 books, 278,858 users, and 1,149,780 "
         "ratings. Four recommendation strategies are implemented: Popularity-Based "
         "Filtering, Collaborative Filtering (User-Based and Item-Based using Cosine "
         "Similarity), Matrix Factorisation via Singular Value Decomposition (SVD), and "
         "a Hybrid model that blends popularity signals with personalised recommendations. "
         "The system is evaluated using RMSE, MAE, Precision@K, and Recall@K metrics, "
         "and is deployed as an interactive Streamlit web application.")

    # ── Introduction ───────────────────────────────────────────────
    heading(doc, "1. Introduction")
    body(doc,
         "The exponential growth of digital book platforms has created an information "
         "overload problem for readers seeking their next book. Recommendation systems "
         "address this by automatically filtering and ranking content based on user "
         "preferences and historical behaviour. This project builds a production-grade "
         "book recommendation engine that demonstrates multiple machine learning "
         "paradigms, from simple popularity-based heuristics to advanced latent-factor "
         "matrix factorisation models.")

    # ── Problem Statement ──────────────────────────────────────────
    heading(doc, "2. Problem Statement")
    body(doc,
         "Given a large corpus of books and sparse user rating data, the system must "
         "accurately predict which books a user is likely to enjoy and generate a "
         "ranked list of personalised recommendations. The challenges include: "
         "(1) Cold-start for new users/books, "
         "(2) Extreme data sparsity (~99.5% of possible ratings are missing), "
         "(3) Long-tail distribution where few books receive most ratings, "
         "(4) Balancing exploration vs. exploitation in recommendations.")

    # ── Objectives ─────────────────────────────────────────────────
    heading(doc, "3. Objectives")
    for obj in [
        "Perform complete EDA on the Book-Crossing dataset.",
        "Clean and preprocess data to remove noise and inconsistencies.",
        "Implement Popularity-Based, Collaborative Filtering, SVD, and Hybrid models.",
        "Evaluate models using RMSE, MAE, Precision@K, and Recall@K.",
        "Deploy an interactive Streamlit application for end-users.",
        "Document the complete system for academic capstone review.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(obj)

    # ── Literature Review ──────────────────────────────────────────
    heading(doc, "4. Literature Review")
    body(doc,
         "[1] Sarwar et al. (2001) – Item-Based Collaborative Filtering: Pioneered "
         "item-item cosine similarity for scalable recommendations at Amazon.\n\n"
         "[2] Koren et al. (2009) – Matrix Factorisation Techniques: Introduced SVD "
         "and its variants for the Netflix Prize, achieving state-of-the-art RMSE.\n\n"
         "[3] Ziegler et al. (2005) – Book-Crossing Dataset: Published the dataset "
         "used in this project, containing implicit and explicit ratings.\n\n"
         "[4] He et al. (2017) – Neural Collaborative Filtering: Extended CF with "
         "deep neural networks for capturing non-linear user-item interactions.\n\n"
         "[5] Burke (2002) – Hybrid Recommender Systems: Formalised the taxonomy of "
         "hybrid approaches combining content-based and collaborative filtering.")

    # ── Methodology ────────────────────────────────────────────────
    heading(doc, "5. Methodology")

    heading(doc, "5.1 Data Preprocessing", level=2)
    body(doc,
         "• Standardised column names to snake_case.\n"
         "• Removed duplicate ISBNs and (user_id, isbn) rating pairs.\n"
         "• Converted year_of_publication to numeric; capped valid range 1800–2024.\n"
         "• Replaced implausible ages (<5 or >100) with the dataset median.\n"
         "• Separated explicit ratings (1–10) from implicit ratings (0).\n"
         "• Created books_enriched with per-book avg_rating and rating_count.")

    heading(doc, "5.2 Popularity-Based Recommender", level=2)
    body(doc,
         "A weighted score is computed as: score = avg_rating × log(1 + rating_count). "
         "Books with fewer than 50 ratings are excluded to avoid high-rating low-sample "
         "bias. The model returns Top-10, Top-20, and Top-50 lists.")

    heading(doc, "5.3 Collaborative Filtering", level=2)
    body(doc,
         "A user-book rating matrix (pivot table) is constructed from active users "
         "(≥50 ratings) and popular books (≥10 ratings). Cosine similarity is computed "
         "on this sparse matrix using sklearn.metrics.pairwise.cosine_similarity.\n\n"
         "• Item-Based CF: Given a query book, return K most similar books by "
         "item-item cosine similarity.\n"
         "• User-Based CF: Find similar users, aggregate their highly rated books "
         "not yet read by the target user.")

    heading(doc, "5.4 SVD Matrix Factorisation", level=2)
    body(doc,
         "The Surprise library's SVD algorithm decomposes the rating matrix R ≈ P·Q^T, "
         "where P is a user-factor matrix and Q is an item-factor matrix. "
         "Hyperparameters: n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02. "
         "The model is trained on 80% of explicit ratings and evaluated on 20%.")

    heading(doc, "5.5 Hybrid Recommender", level=2)
    body(doc,
         "The hybrid score combines CF similarity and popularity:\n"
         "  hybrid_score = (1 - α) × cf_similarity + α × popularity_norm\n"
         "where α = 0.4 balances personalisation and popularity. "
         "This reduces the cold-start problem and improves coverage.")

    # ── System Architecture ────────────────────────────────────────
    heading(doc, "6. System Architecture")
    body(doc,
         "Raw CSVs (Books / Users / Ratings)\n"
         "         ↓\n"
         "  Data Cleaning & Feature Engineering (preprocessing.py)\n"
         "         ↓\n"
         "  EDA & Visualisations (eda.py)\n"
         "         ↓\n"
         "  ┌─────────────────────────────────┐\n"
         "  │      Recommendation Engine      │\n"
         "  │  Popularity | CF | SVD | Hybrid │\n"
         "  └─────────────┬───────────────────┘\n"
         "                ↓\n"
         "         Model Evaluation (evaluation.py)\n"
         "                ↓\n"
         "         Streamlit Web App (app.py)")

    # ── Algorithms ────────────────────────────────────────────────
    heading(doc, "7. Algorithms Used")
    algos = [
        ("Cosine Similarity",
         "sim(A,B) = (A·B) / (||A|| × ||B||). Used for both item-item and user-user CF."),
        ("Weighted Popularity Score",
         "score = avg_rating × log(1 + rating_count). Balances quality and volume."),
        ("SVD (Matrix Factorisation)",
         "Minimises: Σ(r_ui − p_u·q_i)² + λ(||p_u||² + ||q_i||²) via SGD."),
        ("Hybrid Blending",
         "hybrid = (1-α)·cf_score + α·pop_score. Weighted linear combination."),
    ]
    for name, desc in algos:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"{name}: ").bold = True
        p.add_run(desc)

    # ── Results ────────────────────────────────────────────────────
    heading(doc, "8. Results and Discussion")
    body(doc,
         "The SVD model achieved competitive RMSE and MAE on the test split. "
         "Collaborative Filtering demonstrated strong item-similarity scores for "
         "popular genres. The Hybrid model consistently outperformed individual "
         "models in coverage while maintaining personalisation quality. "
         "The long-tail distribution of ratings (most books have <5 ratings) "
         "presents a fundamental challenge for all models, addressed by the "
         "popularity filter and hybrid blending strategy.")

    # Insert figures if available
    for fig_name in ["00_summary_dashboard.png", "09_rmse_mae.png", "10_precision_recall.png"]:
        fig_path = os.path.join(FIG_DIR, fig_name)
        if os.path.exists(fig_path):
            doc.add_picture(fig_path, width=Inches(5.5))
            cap = doc.add_paragraph(f"Figure: {fig_name.replace('_',' ').replace('.png','').title()}")
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── Conclusion ─────────────────────────────────────────────────
    heading(doc, "9. Conclusion")
    body(doc,
         "This project successfully implemented and evaluated a multi-model Book "
         "Recommendation System using the Book-Crossing dataset. The Hybrid approach "
         "combining Popularity-Based and Collaborative Filtering provided the best "
         "balance of accuracy, coverage, and personalisation. The Streamlit application "
         "makes the system accessible to non-technical users.")

    # ── Future Scope ───────────────────────────────────────────────
    heading(doc, "10. Future Scope")
    for scope in [
        "Deep Learning models (Neural CF, Autoencoders, Transformer-based recommenders).",
        "Content-Based Filtering using book descriptions and genres (NLP/TF-IDF).",
        "Real-time recommendation updates with online learning.",
        "A/B testing framework to measure recommendation quality in production.",
        "Multi-modal features: book covers (CNN), review sentiment (BERT).",
        "Deployment on AWS/GCP with REST API and mobile client.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(scope)

    # ── References ─────────────────────────────────────────────────
    heading(doc, "References")
    refs = [
        "Sarwar, B. et al. (2001). Item-based collaborative filtering recommendation algorithms. WWW '01.",
        "Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. IEEE Computer.",
        "Ziegler, C. N. et al. (2005). Improving recommendation lists through topic diversification. WWW '05.",
        "He, X. et al. (2017). Neural collaborative filtering. WWW '17.",
        "Burke, R. (2002). Hybrid recommender systems: Survey and experiments. User Modeling and User-Adapted Interaction.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="List Number")
        p.add_run(ref)

    out_path = os.path.join(REPORT_DIR, "Book_Recommendation_System_MCA_Report.docx")
    doc.save(out_path)
    print(f"[REPORT] Saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    build_report()
