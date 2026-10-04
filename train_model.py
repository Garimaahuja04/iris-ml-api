"""Train a simple Iris classifier and save it to model/iris_model.joblib."""
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).parent / "model" / "iris_model.joblib"


def train() -> float:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "target_names": list(iris.target_names)}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} | test accuracy: {accuracy:.2f}")
    return accuracy


if __name__ == "__main__":
    train()
