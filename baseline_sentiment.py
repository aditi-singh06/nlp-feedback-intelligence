"""
Starter baseline for the proposed NLP feedback project.
Expected CSV columns: text, label
Labels can be positive/negative (or another documented binary classification).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def train_baseline(csv_path: str):
    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str),
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return model


if __name__ == "__main__":
    # Example:
    # train_baseline("data/processed/reviews.csv")
    print("Starter script ready. Add a properly licensed dataset before training.")
