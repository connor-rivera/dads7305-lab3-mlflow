"""Search XGBoost settings, tracking every trial in MLflow."""
import argparse

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score

from src.data import RANDOM_STATE, load_wine_data, split_wine_data
from src.settings import EXPERIMENT_NAME, TRACKING_URI

SEARCH_SPACE = {
    "max_depth": scope.int(hp.quniform("max_depth", 3, 12, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
}


def make_objective(X_train, y_train, X_val, y_val):
    """Build the function Hyperopt calls for each trial."""
    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)

    def objective(params):
        settings = {
            **params,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "seed": RANDOM_STATE,
        }
        with mlflow.start_run(nested=True):
            booster = xgb.train(
                settings,
                train,
                num_boost_round=500,
                evals=[(val, "validation")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
            val_scores = booster.predict(val)
            val_auc = roc_auc_score(y_val, val_scores)

            mlflow.log_params(params)
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_metric("val_auc", val_auc)

            signature = infer_signature(X_train, val_scores)
            mlflow.xgboost.log_model(
                booster,
                name="model",
                signature=signature,
                input_example=X_train.head(5),
            )
        return {"status": STATUS_OK, "loss": -val_auc}

    return objective


def main(max_evals):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_wine_data()
    X_train, X_val, _, y_train, y_val, _ = split_wine_data(data)
    objective = make_objective(X_train, y_train, X_val, y_val)

    with mlflow.start_run(run_name="xgboost_search") as parent:
        fmin(
            fn=objective,
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=Trials(),
            rstate=np.random.default_rng(RANDOM_STATE),
        )
        trials = mlflow.search_runs(
            filter_string=(
                f"tags.mlflow.parentRunId = '{parent.info.run_id}'"
            ),
            order_by=["metrics.val_auc DESC"],
            max_results=1,
        )
        best = trials.iloc[0]
        client = mlflow.MlflowClient()
        client.set_tag(best.run_id, "status", "candidate")
        mlflow.log_metric("best_val_auc", best["metrics.val_auc"])

    print(f"Best trial: {best.run_id}")
    print(f"Best validation AUC: {best['metrics.val_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=32)
    args = parser.parse_args()
    main(args.max_evals)