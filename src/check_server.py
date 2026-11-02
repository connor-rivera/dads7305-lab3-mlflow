"""Send 3 wines to the running model server."""
import requests

from src.data import load_wine_data, split_wine_data

URL = "http://127.0.0.1:5002/invocations"


def main():
    _, _, X_test, _, _, _ = split_wine_data(load_wine_data())
    rows = X_test.head(3).to_dict(orient="split")

    response = requests.post(
        URL, json={"dataframe_split": rows}, timeout=30
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()