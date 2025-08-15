import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Load data
df = pd.read_csv("/data/data.csv")
X = df[['score']]

# List of multipliers
multipliers = [2, 2.5, 3]

for multiplier in multipliers:
    y = df['score'] * multiplier  # target changes each loop

    with mlflow.start_run():
        model = LinearRegression().fit(X, y)

        # Log parameters
        mlflow.log_param("input_feature", "score")
        mlflow.log_param("target_multiplier", multiplier)

        # Log metrics
        mlflow.log_metric("coef", model.coef_[0])
        mlflow.log_metric("intercept", model.intercept_)

        # Save local pickle file
        pickle.dump(model, open(f"/data/model_{multiplier}.pkl", "wb"))

        # Log model to MLflow (with signature & example)
        signature = infer_signature(X, model.predict(X))
        mlflow.sklearn.log_model(
            model,
            name=f"model_multiplier_{multiplier}",  # Use name= for new MLflow versions
            input_example=X.head(1),
            signature=signature
        )

print("Training complete for multipliers:", multipliers)

