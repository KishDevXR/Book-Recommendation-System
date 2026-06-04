# 📚 Book Recommendation System
### MCA Artificial Intelligence Capstone Project

---

## 🗂️ Project Structure

```
Book-Recommendation-System/
│
├── data/                    ← (place copies of raw CSVs here if needed)
├── notebooks/               ← Jupyter Notebook version
├── models/                  ← Saved pickle files for all trained models
├── outputs/
│   ├── figures/             ← All EDA & evaluation plots (PNG)
│   ├── books_clean.csv
│   ├── users_clean.csv
│   ├── explicit_ratings.csv
│   ├── books_enriched.csv
│   └── model_comparison.csv
├── reports/                 ← Final MCA report (DOCX / PDF)
├── src/
│   ├── preprocessing.py     ← Data loading, cleaning, feature engineering
│   ├── eda.py               ← 10 professional EDA visualisations
│   ├── recommender.py       ← All 4 recommendation engines
│   ├── evaluation.py        ← RMSE, MAE, Precision@K, Recall@K
│   └── main.py              ← End-to-end pipeline runner
│
├── app.py                   ← Streamlit web application
├── generate_report.py       ← MCA DOCX report generator
├── requirements.txt
└── README.md
```

---

## 📦 Datasets

| File          | Description                        | Size         |
|---------------|------------------------------------|--------------|
| `Books.csv`   | 271,360 books with metadata        | ~70 MB       |
| `Users.csv`   | 278,858 users with location & age  | ~10 MB       |
| `Ratings.csv` | 1,149,780 ratings (0 = implicit)   | ~22 MB       |

Place the raw CSV files in the `../dataset/` folder relative to this project  
(i.e., `d:\MCA\project\dataset\`).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
cd Book-Recommendation-System
python src/main.py
```
This will:
- Clean all datasets
- Generate 12+ EDA figures
- Train Popularity, CF, SVD, and Hybrid models
- Evaluate and print a comparison table
- Save all models and outputs

### 3. Launch the Streamlit App
```bash
cd Book-Recommendation-System
python -m streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501)

### 4. Generate the MCA Report
```bash
python generate_report.py
```
Output: `reports/Book_Recommendation_System_MCA_Report.docx`

---

## 🤖 Recommendation Models

| Model                     | Technique                        | Personalised |
|---------------------------|----------------------------------|:------------:|
| **Popularity-Based**      | Weighted Score (avg × log count) | ❌            |
| **Item-Based CF**         | Cosine Similarity (book-user matrix) | ✅        |
| **User-Based CF**         | Cosine Similarity (user-book matrix) | ✅        |
| **SVD (Matrix Factorisation)** | Surprise Library SVD        | ✅            |
| **Hybrid**                | Popularity + CF blended (α=0.4)  | ✅            |

---

## 📊 Key Functions

```python
# Recommend books similar to a given title
recommend_books("Harry Potter and the Sorcerer's Stone")

# Personalised recommendations for a user
recommend_for_user(276726)
```

**Example Output:**
```
═══════════════════════════════════════════════════════
  Book Recommendations for: 'Harry Potter and the Sorcerer's Stone'
═══════════════════════════════════════════════════════
   1. Harry Potter and the Chamber of Secrets
       Author : J. K. Rowling
   2. Harry Potter and the Prisoner of Azkaban
       Author : J. K. Rowling
   3. The Lord of the Rings
       Author : J. R. R. Tolkien
  ...
```

---

## 🏗️ System Architecture

```
Books.csv ──┐
Users.csv ──┼──► Data Loading & Cleaning ──► Feature Engineering
Ratings.csv ┘              │
                           ▼
              ┌────────────────────────────┐
              │   Recommendation Engine    │
              ├──────────┬─────────┬───────┤
              │Popularity│   CF   │  SVD  │
              └──────────┴────┬────┴───────┘
                              │
                              ▼
                    Hybrid Recommender
                              │
                              ▼
                   Final Recommendations
                              │
                    ┌─────────┴──────────┐
                    │  Streamlit Web App  │
                    └─────────────────────┘
```

---

## 📈 Evaluation Metrics

| Metric          | Description                                      |
|-----------------|--------------------------------------------------|
| **RMSE**        | Root Mean Square Error (SVD rating prediction)   |
| **MAE**         | Mean Absolute Error (SVD rating prediction)      |
| **Precision@K** | Fraction of top-K recommendations that are relevant |
| **Recall@K**    | Fraction of relevant items in top-K              |
| **Coverage**    | Fraction of books the model can recommend for    |

---

## 🎨 Streamlit App Features

| Page                  | Feature                                         |
|-----------------------|-------------------------------------------------|
| 🏠 Dashboard           | KPIs, Rating distribution, Decade chart, Top 10 |
| 🔍 Search & Explore    | Full-text search by title/author                |
| 📖 Similar Books       | Item-Based / Hybrid recommendations              |
| 👤 User Recommendations| Personalised user-based recommendations         |
| 📊 Model Evaluation    | Comparison table, all 12 saved visualisations   |

---

## 👩‍💻 Author & Course

- **Project Type:** MCA Artificial Intelligence Capstone
- **Dataset:** Book-Crossing (Cai-Nicolas Ziegler, 2004)
- **Libraries:** Pandas · NumPy · Matplotlib · Seaborn · Scikit-Learn · Surprise · Streamlit
