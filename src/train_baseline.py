"""Train a baseline random forest and track it with MLflow."""
import argparse
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data import RANDOM_STATE, load_wine_data, split_wine_data
from src.settings import EXPERIMENT_NAME, TRACKING_URI

N_TREES = 10

class ProbabilityModel(mlflow.pyfunc.PythonModel):
    """Make predict() return the chance a wine is good."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

def main(n_trees):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_wine_data()
    X_train, X_val, _, y_train, y_val, _ = split_wine_data(data)

    with mlflow.start_run(run_name="baseline_random_forest") as run:
        forest = RandomForestClassifier(
            n_estimators=n_trees, random_state=RANDOM_STATE
        )
        forest.fit(X_train, y_train)

        val_scores = forest.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_scores)
        mlflow.set_tag("status", "candidate")

        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("n_estimators", n_trees)
        mlflow.log_metric("val_auc", val_auc)

        signature = infer_signature(X_train, val_scores)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=ProbabilityModel(forest),
            signature=signature,
            input_example=X_train.head(5),
        )

    print(f"Run ID: {run.info.run_id}")
    print(f"Validation AUC: {val_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trees", type=int, default=N_TREES)
    args = parser.parse_args()
    main(args.n_trees)