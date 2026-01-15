import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data.csv")
X = data[["feature1", "feature2"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Experiment name
mlflow.set_experiment("LinearRegression_Experiment")

# Different hyperparameter values
params_list = [
    {"fit_intercept": True},
    {"fit_intercept": False}
]

for params in params_list:
    with mlflow.start_run():
        model = LinearRegression(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log parameters
        mlflow.log_param("fit_intercept", params["fit_intercept"])

        # Log metrics
        mlflow.log_metric("mse", mse)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"Run completed | Params: {params} | MSE: {mse}")
