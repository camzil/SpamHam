# This model trains scikit learn neural network MLPClassifier
# on word2vec vectors and evaluates the accuracy of the model

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load vectors
def load_vectors(data_folder="."):
    X_train = np.load(f"{data_folder}/X_train_w2v.npy")
    X_dev   = np.load(f"{data_folder}/X_dev_w2v.npy")
    X_test  = np.load(f"{data_folder}/X_test_w2v.npy")

    y_train = np.load(f"{data_folder}/y_train.npy")
    y_dev   = np.load(f"{data_folder}/y_dev.npy")
    y_test  = np.load(f"{data_folder}/y_test.npy")

    print("Train shape:", X_train.shape)
    print("Dev shape:", X_dev.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Train MLP
def train_mlp(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), # 2 hidden layers: 1st layer has 64 neurons and 2nd layer has 32 neurons
        activation='relu', # Activation function
        max_iter=200, # Max number of iterations
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Evaluate
def evaluate(model, X, y, name):
    y_pred = model.predict(X) # Predicts labels for each message

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y, y_pred): .4f}")
    print(classification_report(y, y_pred, target_names=["ham", "spam"]))
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print("          Pred Ham   Pred Spam")
    print(f"True Ham     {cm[0][0]}         {cm[0][1]}")
    print(f"True Spam    {cm[1][0]}         {cm[1][1]}")


# Load vectors
X_train, X_dev, X_test, y_train, y_dev, y_test = load_vectors(".")

# Train model
model = train_mlp(X_train, y_train)

# Evaluate on dev and test Sets
evaluate(model, X_dev, y_dev, "Dev")
evaluate(model, X_test, y_test, "Test")