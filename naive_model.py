# SMS Spam / Ham Classifier — Naive Bayes
import numpy as np
import scipy.sparse as sp

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load features saved by ETL pipeline
# The ETL pipeline already cleaned the text, split into 60/20/20, and built Bag-of-Words matrices. We just load those files here.
def load_features(data_folder="."):
    print("Loading pre-built features from ETL pipeline...")

    # Load Bag-of-Words matrices (built by CountVectorizer in ETL Section 4A)
    X_train = sp.load_npz(f"{data_folder}/X_train_bow.npz")
    X_dev = sp.load_npz(f"{data_folder}/X_dev_bow.npz")
    X_test = sp.load_npz(f"{data_folder}/X_test_bow.npz")

    # Load labels — 0 = ham (safe), 1 = spam
    y_train = np.load(f"{data_folder}/y_train.npy")
    y_dev = np.load(f"{data_folder}/y_dev.npy")
    y_test = np.load(f"{data_folder}/y_test.npy")

    print(f"\nLoaded successfully!")
    print(f"  Train : {X_train.shape}  spam rate: {y_train.mean():.1%}")
    print(f"  Dev   : {X_dev.shape}    spam rate: {y_dev.mean():.1%}")
    print(f"  Test  : {X_test.shape}   spam rate: {y_test.mean():.1%}")

    return X_train, X_dev, X_test, y_train, y_dev, y_test


# Train Multinomial Naive Bayes
# Naive Bayes learns how often each word appears in spam vs ham words like "free", "winner", "prize" appear more in spam and ords like "lunch", "home", "tomorrow" appear more in ham.
# For each new message it calculates the probability of spam vs ham based on the words it contains, and picks the higher one.
# I use MultinomialNB because it is designed for word count datawhich is exactly what Bag-of-Words gives us.
# I also used alpha=1.0 is Laplace smoothing to prevents zero probability errors when a word in the test set was never seen during training.

def train_model(X_train, y_train):
    print("\nTraining Multinomial Naive Bayes model...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


# Evaluate the model
# We evaluate on Dev set first, then on Test set.
def evaluate_model(model, X, y_true, set_name="Dev"):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["ham", "spam"])

    print("\n" + "=" * 55)
    print(f"  MODEL EVALUATION — {set_name.upper()} SET")
    print("=" * 55)
    print(f"\nOverall Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")

    print("\nConfusion Matrix:")
    print(f"  {'':25s}  Predicted Ham  Predicted Spam")
    print(f"  {'Actual Ham':25s}  {cm[0, 0]:>13}  {cm[0, 1]:>14}")
    print(f"  {'Actual Spam':25s}  {cm[1, 0]:>13}  {cm[1, 1]:>14}")

    print("\nWhat each cell means:")
    print(
        f"  True Negatives  (ham  -> ham)  : {cm[0, 0]:>4}  correctly left alone")
    print(
        f"  False Positives (ham  -> spam) : {cm[0, 1]:>4}  normal message flagged!")
    print(
        f"  False Negatives (spam -> ham)  : {cm[1, 0]:>4}  spam slipped through!")
    print(
        f"  True Positives  (spam -> spam) : {cm[1, 1]:>4}  spam caught correctly")

    print("\nClassification Report:")
    print(report)
    print("=" * 55)


# MAIN pipeline
def main():
    print("=" * 55)
    print("  SpamHam — Naive Bayes SMS Classifier")
    print("  NLP Team Project")
    print("=" * 55)

    # Step 1: Load ETL pipeline features
    print("\n[Step 1] Loading ETL pipeline features...")
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_features(".")

    # Step 2: Train the model
    print("\n[Step 2] Training Naive Bayes model...")
    model = train_model(X_train, y_train)

    # Step 3: Check results on Dev set
    print("\n[Step 3] Evaluating on Dev set...")
    evaluate_model(model, X_dev, y_dev, set_name="Dev")

    # Step 4: Final score on Test set
    print("\n[Step 4] Final evaluation on Test set...")
    evaluate_model(model, X_test, y_test, set_name="Test")

    # ETL reference: Logistic Regression on BoW got Accuracy 97%, Spam F1 0.90
    print("\n[Reference] ETL sanity check (Logistic Regression on BoW):")
    print("  Accuracy: 97%  |  Spam F1: 0.90")


if __name__ == "__main__":
    main()
