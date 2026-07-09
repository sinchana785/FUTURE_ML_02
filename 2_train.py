import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

if __name__ == "__main__":
    print("📦 Loading clean preprocessed production data...")
    df = pd.read_csv("cleaned_tickets.csv").dropna()

    X = df['cleaned_text']
    y_cat = df['category']
    y_prio = df['priority']

    X_train, X_test, y_train_cat, y_test_cat, y_train_prio, y_test_prio = train_test_split(
        X, y_cat, y_prio, test_size=0.20, random_state=42
    )

    print("📐 Vectorizing features via TF-IDF...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("🧠 Training Category Classifier (Optimized Linear Solver)...")
    cat_model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    cat_model.fit(X_train_tfidf, y_train_cat)

    print("🧠 Training Priority Classifier (Optimized Linear Solver)...")
    prio_model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    prio_model.fit(X_train_tfidf, y_train_prio)

    print("💾 Freezing real-world trained model artifacts to disk...")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(cat_model, "category_model.pkl")
    joblib.dump(prio_model, "priority_model.pkl")

    print("\n" + "="*60)
    print("📊 CATEGORY CLASSIFICATION PERFORMANCE REPORT")
    print("="*60)
    cat_preds = cat_model.predict(X_test_tfidf)
    print(classification_report(y_test_cat, cat_preds))

    print("\n" + "="*60)
    print("📊 PRIORITY CLASSIFICATION PERFORMANCE REPORT")
    print("="*60)
    prio_preds = prio_model.predict(X_test_tfidf)
    print(classification_report(y_test_prio, prio_preds))