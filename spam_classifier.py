"""SMS/email spam classifier: TF-IDF + Multinomial Naive Bayes.

Train:    python spam_classifier.py
Predict:  python spam_classifier.py --predict "Congratulations! You won a free iPhone."
"""

import argparse
import string
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/spam.csv")
MODEL_PATH = Path("spam_model.joblib")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def load_stopwords():
    """NLTK English stopwords, falling back to scikit-learn's list if unavailable."""
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except Exception:
        print("NLTK stopwords unavailable, using scikit-learn's stopword list.")
        return set(ENGLISH_STOP_WORDS)


STOPWORDS = load_stopwords()


def clean_text(text):
    """Lowercase, strip punctuation, drop stopwords."""
    words = str(text).lower().translate(PUNCT_TABLE).split()
    return " ".join(w for w in words if w not in STOPWORDS)


def load_data(path=DATA_PATH):
    if not path.exists():
        sys.exit(f"Dataset not found at {path}. Place spam.csv (SMS Spam Collection) in data/.")
    data = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    data.columns = ["label", "message"]
    data = data.dropna().drop_duplicates().reset_index(drop=True)
    print(data.head())
    print(data["label"].value_counts())
    return data


def train():
    data = load_data()
    data["clean"] = data["message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("nb", MultinomialNB()),
    ])
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    print("First 10 predictions:", prediction[:10])
    print("Accuracy:", round(accuracy_score(y_test, prediction), 4))
    print(classification_report(y_test, prediction))
    cm = confusion_matrix(y_test, prediction, labels=model.classes_)
    print(cm)

    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(cmap="Blues")
    plt.title("Spam Classifier - Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def predict(message, model=None):
    if model is None:
        if not MODEL_PATH.exists():
            sys.exit("No saved model found. Run `python spam_classifier.py` first to train one.")
        model = joblib.load(MODEL_PATH)
    return model.predict([clean_text(message)])[0]


def main():
    parser = argparse.ArgumentParser(description="Spam classifier")
    parser.add_argument("--predict", metavar="TEXT", help="classify a message with the saved model")
    args = parser.parse_args()

    if args.predict:
        print("Prediction:", predict(args.predict).title())
        return

    model = train()
    sample = "Congratulations! You have won a free iPhone. Click here to claim your prize."
    print(f"\nSample: {sample}")
    print("Prediction:", predict(sample, model).title())


if __name__ == "__main__":
    main()
