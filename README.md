# Spam Classifier

Classifies messages as **spam** or **ham** (not spam) using TF-IDF features and a Multinomial Naive Bayes model, trained on the SMS Spam Collection dataset.

## Pipeline

1. Load `spam.csv`, keep the label and message columns, drop duplicates
2. Clean text: lowercase, remove punctuation, remove English stopwords
3. TF-IDF vectorization (top 5,000 features)
4. Stratified 80/20 train/test split
5. Train Multinomial Naive Bayes
6. Evaluate: accuracy, precision, recall, F1, confusion matrix
7. Save the whole pipeline (vectorizer + model) to `spam_model.joblib`

## Project Structure

```
spam-classifier/
├── data/                  # put spam.csv here (not included)
├── spam_classifier.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset

Download the SMS Spam Collection (e.g. "SMS Spam Collection Dataset" on Kaggle) and save it as `data/spam.csv`. Columns used: `v1` (label: ham/spam) and `v2` (message text).

## Installation

```bash
git clone https://github.com/<your-username>/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
```

## Usage

Train, evaluate and save the model:

```bash
python spam_classifier.py
```

Classify your own message with the saved model:

```bash
python spam_classifier.py --predict "Congratulations! You won a free iPhone. Click here to claim."
```

## Notes

- NLTK stopwords are downloaded on first run; if that fails the script falls back to scikit-learn's stopword list.
- The saved `.joblib` file should only be loaded from sources you trust.
- The dataset is SMS text, so accuracy on long emails may be lower.

## License

MIT
