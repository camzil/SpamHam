import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

import etl


RANDOM_STATE = 42


def build_tfidf_features(x_train, x_dev, x_test, max_features: int = 5000):
    """Turn the cleaned ETL text into TF-IDF feature matrices."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_dev_tfidf = vectorizer.transform(x_dev)
    x_test_tfidf = vectorizer.transform(x_test)

    print(f"TF-IDF train shape: {x_train_tfidf.shape}")
    return x_train_tfidf, x_dev_tfidf, x_test_tfidf, vectorizer


def build_dense_nn(hidden_units: int = 64) -> MLPClassifier:
    """Build a dense neural network for TF-IDF features."""
    return MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=80,
        #prevent overfitting
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
        random_state=RANDOM_STATE,
    )


def evaluate(model: MLPClassifier, x, y, split_name: str) -> float:
    """Print the main evaluation numbers, especially spam F1."""
    y_pred = model.predict(x)
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
    # ETL.py gives us cleaned train/dev/test text. This script handles TF-IDF.
    x_train_tfidf, x_dev_tfidf, x_test_tfidf, tfidf_vectorizer = build_tfidf_features(
        ETL.X_train,
        ETL.X_dev,
        ETL.X_test,
        max_features=5000,
    )

    model = build_dense_nn(hidden_units=64)

    print("Training dense neural network on TF-IDF features...")
    model.fit(x_train_tfidf, ETL.y_train)
    print("Training complete.\n")

    # Dev is for checking the model while still experimenting.
    evaluate(model, x_dev_tfidf, ETL.y_dev, "Dev")

    # Test is the final held-out evaluation.
    print("Final held-out test evaluation:")
    evaluate(model, x_test_tfidf, ETL.y_test, "Test")

    with open("tfidf_dense_nn.pkl", "wb") as model_file:
        pickle.dump(
            {
                "model": model,
                "tfidf_vectorizer": tfidf_vectorizer,
                "label_encoder": ETL.label_encoder,
            },
            model_file,
        )
    print("Saved trained model, TF-IDF vectorizer, and label encoder to tfidf_dense_nn.pkl")


if __name__ == "__main__":
    main()
