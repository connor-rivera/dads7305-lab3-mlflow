"""Promote the newest candidate only if it beats the champion."""
import mlflow
from sklearn.metrics import roc_auc_score

from src.data import load_wine_data, split_wine_data
from src.settings import EXPERIMENT_NAME, MODEL_NAME, TRACKING_URI

MIN_IMPROVEMENT = 0.005


def find_candidate():
    """Return the newest run still waiting for a decision."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.status = 'candidate'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return None if runs.empty else runs.iloc[0]


def score(model_uri, X_test, y_test):
    """Return a registered model's AUC on the test set."""
    model = mlflow.pyfunc.load_model(model_uri)
    return roc_auc_score(y_test, model.predict(X_test))


def get_champion_version(client):
    """Return the champion version, or None if there isn't one."""
    try:
        version = client.get_model_version_by_alias(
            MODEL_NAME, "champion"
        )
    except mlflow.exceptions.MlflowException:
        return None
    return version.version


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    candidate = find_candidate()
    if candidate is None:
        print("No new candidate to review.")
        return

    run_id = candidate.run_id
    _, _, X_test, _, _, y_test = split_wine_data(load_wine_data())

    version = mlflow.register_model(
        f"runs:/{run_id}/model", MODEL_NAME
    ).version
    client.set_registered_model_alias(
        MODEL_NAME, "challenger", version
    )
    challenger_auc = score(
        f"models:/{MODEL_NAME}@challenger", X_test, y_test
    )

    champion = get_champion_version(client)
    if champion is None:
        champion_auc = None
        promote = True
        reason = "no champion yet"
    else:
        champion_auc = score(
            f"models:/{MODEL_NAME}@champion", X_test, y_test
        )
        gain = challenger_auc - champion_auc
        promote = gain >= MIN_IMPROVEMENT
        reason = f"gain {gain:+.4f}, needed {MIN_IMPROVEMENT:+.4f}"

    if promote:
        client.set_registered_model_alias(
            MODEL_NAME, "champion", version
        )
        client.delete_registered_model_alias(
            MODEL_NAME, "challenger"
        )

    decision = "promoted" if promote else "rejected"
    client.set_tag(run_id, "status", decision)
    client.log_metric(run_id, "test_auc", challenger_auc)
    client.set_model_version_tag(
        MODEL_NAME, version, "decision", decision
    )
    client.set_model_version_tag(
        MODEL_NAME, version, "test_auc", f"{challenger_auc:.4f}"
    )

    print(f"Candidate: version {version}")
    print(f"Candidate test AUC: {challenger_auc:.4f}")
    if champion_auc is not None:
        print(f"Champion test AUC: {champion_auc:.4f} "
              f"(version {champion})")
    print(f"Decision: {decision.upper()} ({reason})")


if __name__ == "__main__":
    main()