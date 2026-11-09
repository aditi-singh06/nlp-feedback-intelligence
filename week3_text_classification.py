"""
Week 3: Text Classification Model Implementation
Project: NLP Feedback Intelligence
This reproducible controlled demo uses a small synthetic review dataset.
It is NOT a real-world benchmark. Replace the demo data with a licensed
public dataset (e.g., IMDb reviews or a suitable Hugging Face/Kaggle dataset)
for a full-scale experiment.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

DATA = [
("The product is excellent and works perfectly","positive"),
("I am very happy with this purchase","positive"),
("Amazing quality and fast delivery","positive"),
("The service was helpful and friendly","positive"),
("Great value for money and easy to use","positive"),
("I love this product, it is wonderful","positive"),
("Very satisfied with the quality","positive"),
("The app is simple, useful and reliable","positive"),
("Delivery was quick and packaging was good","positive"),
("Customer support solved my issue quickly","positive"),
("This is a fantastic product","positive"),
("Works exactly as described, highly recommended","positive"),
("The quality is good and performance is excellent","positive"),
("A smooth experience from order to delivery","positive"),
("I would happily buy this again","positive"),
("The product is terrible and stopped working","negative"),
("I am disappointed with this purchase","negative"),
("Poor quality and very slow delivery","negative"),
("Customer service was rude and unhelpful","negative"),
("Waste of money, not worth buying","negative"),
("I hate this product, it is awful","negative"),
("Very dissatisfied with the poor quality","negative"),
("The app is confusing, slow and unreliable","negative"),
("Delivery was late and packaging was damaged","negative"),
("Support ignored my complaint","negative"),
("This is a horrible product","negative"),
("It does not work as described, avoid it","negative"),
("The quality is bad and performance is terrible","negative"),
("A frustrating experience from order to delivery","negative"),
("I would never buy this again","negative"),
]
df = pd.DataFrame(DATA, columns=["text", "label"])
X_train, X_test, y_train, y_test = train_test_split(
    df.text, df.label, test_size=0.30, random_state=42, stratify=df.label
)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced")
}
for name, classifier in models.items():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1,
                                  strip_accents="unicode")),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    print(f"\\n{name}")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall:    {r:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print("Confusion matrix (labels: negative, positive):")
    print(confusion_matrix(y_test, pred, labels=["negative", "positive"]))
    print(classification_report(y_test, pred, zero_division=0))

print("\\nNote: results on this tiny synthetic dataset are illustrative only.")
