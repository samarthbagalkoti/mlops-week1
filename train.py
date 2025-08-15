import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def get_env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    return val


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_mlflow_url(base: str, experiment_id: str, run_id: str) -> str | None:
    # Only build links for http(s) URIs
    if not base or not (base.startswith("http://") or base.startswith("https://")):
        return None
    # Normalize (MLflow UI path)
    base = base.rstrip("/")
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def main() -> None:
    # ---- Config via env ----
    DATA = get_env("DATA_PATH", "/data/data.csv")
    MODEL_DIR = get_env("MODEL_PATH", "/data")  # treated as directory
    TRACKING = get_env("MLFLOW_TRACKING_URI")   # e.g., http://host.docker.internal:8080
    EXPERIMENT = get_env("MLFLOW_EXPERIMENT", "Default")

    # Optionally override multipliers via env: e.g. "2,2.5,3"
    raw_mults = get_env("MULTIPLIERS", "2,2.5,3")
    multipliers = [float(x.strip()) for x in raw_mults.split(",") if x.strip()]

    # ---- MLflow setup ----
    if TRACKING:
        mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENT)

    # ---- Data load ----
    df = pd.read_csv(DATA)
    df.columns = [c.strip().lower() for c in df.columns]
    col = "score" if "score" in df.columns else df.columns[0]

    # Use a single-feature demo X and dummy y = multiplier * X
    # Cast to float to avoid MLflow schema warnings with ints & NaNs
    X = df[[col]].astype("float64")

    # ---- Ensure model output dir ----
    ensure_dir(MODEL_DIR)

    for m in multipliers:
        safe = str(m).replace(".", "_")  # for filenames and MLflow model "name"
        y = (df[col].astype("float64")) * m

        # Train
        model = LinearRegression().fit(X, y)

        # Save to disk (mounted volume)
        model_path = Path(MODEL_DIR) / f"model_x{safe}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Trained with multiplier {m} and saved to {model_path}")

        # Log to MLflow
        with mlflow.start_run(run_name=f"multiplier_{safe}") as run:
            mlflow.log_param("multiplier", m)
            mlflow.log_metric("r2_score_on_train", model.score(X, y))

            # Signature & example
            signature = infer_signature(X, y)
            mlflow.sklearn.log_model(
                sk_model=model,
                name=f"model_x{safe}",       # use 'name' (no dots allowed)
                signature=signature,
                input_example=X.head(2),
            )

            # Nice links (if TRACKING is http/s)
            url = build_mlflow_url(TRACKING or "", run.info.experiment_id, run.info.run_id)
            if url:
                print(f"🏃 View run multiplier_{safe} at: {url}")

    # Optional: also print experiment link (if TRACKING is http/s)
    if TRACKING and (TRACKING.startswith("http://") or TRACKING.startswith("https://")):
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT)
        if exp:
            print(f"🧪 View experiment at: {TRACKING.rstrip('/')}/#/experiments/{exp.experiment_id}")


if __name__ == "__main__":
    main()

