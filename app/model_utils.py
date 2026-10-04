"""Loads the trained model and exposes a prediction function."""
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.joblib"

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run: python train_model.py"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_species(sepal_length: float, sepal_width: float,
                    petal_length: float, petal_width: float) -> str:
    bundle = _load()
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    index = int(bundle["model"].predict(features)[0])
    return bundle["target_names"][index]
