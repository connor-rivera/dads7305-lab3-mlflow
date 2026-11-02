"""Score a batch of wines with the current champion."""
from pathlib import Path

import mlflow

from src.data import load_wine_data, split_wine_data
from src.settings import MODEL_NAME, TRACKING_URI

OUTPUT_FILE = Path("outputs/batch_scores.csv")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

    _, _, X_test, _, _, _ = split_wine_data(load_wine_data())
    scores = X_test.copy()
    scores["good_wine_chance"] = model.predict(X_test)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    scores.to_csv(OUTPUT_FILE, index=False)
    print(f"Scored {len(scores)} wines")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()