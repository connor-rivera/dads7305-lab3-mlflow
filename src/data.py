"""Load the wine data and split it the same way every run."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 123
GOOD_WINE_SCORE = 7

def load_wine_data():
    """Return one table of red and white wines."""
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=",")

    white["is_red"] = 0
    red["is_red"] = 1

    data = pd.concat([red, white], axis=0, ignore_index=True)
    data.columns = [col.replace(" ", "_") for col in data.columns]
    return data

def split_wine_data(data):
    """Split into 60% train, 20% validation, 20% test."""
    data = data.drop_duplicates()
    X = data.drop(columns=["quality"])
    y = (data["quality"] >= GOOD_WINE_SCORE).astype(int)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=0.6, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5,
        random_state=RANDOM_STATE, stratify=y_rest
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    wine = load_wine_data()
    splits = split_wine_data(wine)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    print(f"Rows loaded: {len(wine)}")
    print(f"Exact duplicates: {wine.duplicated().sum()}")
    print(f"Train / val / test: "
          f"{len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"Good wine share: {y_train.mean():.3f}")