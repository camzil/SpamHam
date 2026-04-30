import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier


RANDOM_STATE = 42


def load_tfidf_files():
    """
    Load the TF-IDF files that were already created by the ETL notebook/script.
    This file does not import etl.py or use the Logistic Regression sanity check.
    """
    X_train = sp.load_npz("X_train_tfidf.npz")
    X_dev = sp.load_npz("X_dev_tfidf.npz")
    X_test = sp.load_npz("X_test_tfidf.npz")

    y_train = np.load("y_train.npy")
    y_dev = np.load("y_dev.npy")
    y_test = np.load("y_test.npy")

    print("Loaded saved TF-IDF data:")
    print(f"  Train: {X_train.shape}")
    print(f"  Dev:   {X_dev.shape}")
    print(f"  Test:  {X_test.shape}")

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def build_dense_nn(hidden_units: int = 64) -> MLPClassifier:
    """Build a small neural network for the saved TF-IDF features."""
    return MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=80,
        # This helps stop the model before it starts memorizing the training set.
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
        random_state=RANDOM_STATE,
    )


def evaluate(model: MLPClassifier, X, y, split_name: str) -> float:
    """Print the main evaluation numbers for one split."""
    y_pred = model.predict(X)
    spam_f1 = f1_score(y, y_pred, pos_label=1)

    print("=" * 58)
    print(f"{split_name} evaluation")
    print("=" * 58)
    print(classification_report(y, y_pred, target_names=["ham", "spam"], digits=4))

    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix:")
    print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}")
    print(f"Spam F1: {spam_f1:.4f}\n")
    return spam_f1


def main() -> None:
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_tfidf_files()

    model = build_dense_nn(hidden_units=64)

    print("\nTraining dense neural network on saved TF-IDF features...")
    model.fit(X_train, y_train)
    print("Training complete.\n")

    evaluate(model, X_dev, y_dev, "Dev")

    print("Final held-out test evaluation:")
    evaluate(model, X_test, y_test, "Test")

    with open("tfidf_dense_nn.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    print("Saved trained neural network to tfidf_dense_nn.pkl")


if __name__ == "__main__":
    main()
