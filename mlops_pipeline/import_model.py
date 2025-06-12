# In file: mlops_pipeline/import_model.py

import mlflow
import bentoml

MLFLOW_TRACKING_URI = "/home/codespace/.config/zenml/local_stores/d2a56ecc-8f9e-4075-a770-ef2c192e4756"
REGISTERED_MODEL_NAME = "plot_price_bento_model"

print(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# This URI asks the MLflow Model Registry for the latest version of our named model.
model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"

print(f"Loading model from Model Registry URI: {model_uri}")
loaded_model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully from MLflow Model Registry.")

# Save the loaded model into the local BentoML model store
print("Saving model to BentoML model store...")
bento_model = bentoml.sklearn.save_model(REGISTERED_MODEL_NAME, loaded_model)
print(f"Model '{bento_model.tag}' saved to BentoML store.")