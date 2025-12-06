"""Prediccion script for the MLflow model.

This script loads a trained model from MLflow and makes predictions on new data.

$ python make_predictions.py
"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
X = df.drop(columns=["quality"])

## debe verificarse el run_id del modelo que se quiere cargar
## se pude obtener el run_id desde la UI de MLflow

logged_model = "runs:/ebe1519ada4d4df299276cc52591c309/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(X)

print(y)