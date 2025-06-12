# In file: mlops_pipeline/run_pipeline.py

import os
import pandas as pd
import mlflow
import bentoml
import xgboost as xgb
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline as SklearnPipeline
from zenml import step, pipeline

# --- 1. Define ZenML Steps ---

@step
def load_and_clean_data() -> pd.DataFrame:
    """Loads data and performs initial cleaning."""
    print("Loading and cleaning data...")
    DATA_PATH = os.path.join("data", "final_corrected_augmented_dataset.csv")
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=['Price', 'Area', 'Location', 'density', 'beach_proximity', 'lake_proximity'], inplace=True)
    return df

@step
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering logic."""
    print("Engineering features...")
    
    def add_weighted_features(data):
        data['Weighted_Beachfront'] = (data['beach_proximity'] == 'Beachfront').astype(int) * 2.5
        data['Weighted_Seaview'] = (data['beach_proximity'] == 'Sea view').astype(int) * 2.0
        data['Weighted_Lakefront'] = (data['lake_proximity'] == 'Lakefront').astype(int) * 1.8
        data['Weighted_Lakeview'] = (data['lake_proximity'] == 'Lake view').astype(int) * 1.5
        return data

    def add_location_mean_price(data):
        data['Price_per_cent'] = data['Price'] / data['Area']
        mean_price_per_location = data.groupby("Location")['Price_per_cent'].mean().rename("Mean_Price_per_Cent").reset_index()
        data = pd.merge(data, mean_price_per_location, on="Location", how="left")
        data['Mean_Price_per_Cent'].fillna(data['Price_per_cent'].mean(), inplace=True)
        return data

    def add_area_density_feature(data):
        data['Area_Density'] = data['Area'] * (data['density'] == 'High').astype(int)
        return data

    df = add_weighted_features(df)
    df = add_location_mean_price(df)
    df = add_area_density_feature(df)
    return df

@step(experiment_tracker="mlflow_tracker")
def train_and_save_to_bentoml(df: pd.DataFrame):
    """Trains the model, logs metrics to MLflow, and saves the final model to the BentoML local store."""
    print("Starting model training...")
    
    features = [
        'Area', 'Mean_Price_per_Cent', 'Weighted_Beachfront', 'Weighted_Seaview',
        'Weighted_Lakefront', 'Weighted_Lakeview', 'Area_Density', 'Location'
    ]
    target = 'Price'

    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    encoder = ce.TargetEncoder(cols=['Location'])
    
    # We now put the encoder inside the Scikit-learn pipeline
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=150,
        learning_rate=0.05, max_depth=4, seed=42
    )
    
    full_pipeline = SklearnPipeline(steps=[
        ('encoder', encoder),
        ('regressor', model)
    ])

    # Train the complete pipeline
    full_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate and log metrics to MLflow
    predictions = full_pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mlflow.log_metric("rmse", rmse)
    print(f"Logged metric to MLflow RMSE: {rmse:,.2f}")
    
    # Save the trained pipeline to the BentoML local model store
    print("Saving model to BentoML store...")
    bento_model = bentoml.sklearn.save_model(
        name="plot_price_bento_model", # This is the name we'll use in our service
        model=full_pipeline,
        signatures={"predict": {"batchable": True, "batch_dim": 0}},
    )
    print(f"Model saved to BentoML store: {bento_model.tag}")


# --- 2. Define the ZenML Pipeline ---
@pipeline
def plot_price_train_and_save_pipeline():
    """Defines the sequence of steps to train and save the model."""
    df_cleaned = load_and_clean_data()
    df_featured = engineer_features(df=df_cleaned)
    train_and_save_to_bentoml(df=df_featured)

# --- 3. Run the Pipeline ---
if __name__ == "__main__":
    print("Starting ZenML pipeline to train and save model...")
    plot_price_train_and_save_pipeline()
    print("Pipeline execution finished successfully.")