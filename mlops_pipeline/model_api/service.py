# In file: mlops_pipeline/model_api/service.py

import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON

# --- Load data ONCE at startup ---
TRAINING_DF = pd.read_csv("data/final_corrected_augmented_dataset.csv")

# --- Get the model from the local BentoML store ---
model_runner = bentoml.models.get(
    "plot_price_bento_model:latest"
).to_runner()

svc = bentoml.Service("plot_price_predictor", runners=[model_runner])

# --- Feature Engineering Functions ---
def add_weighted_features(data: pd.DataFrame) -> pd.DataFrame:
    data['Weighted_Beachfront'] = (data['beach_proximity'] == 'Beachfront').astype(int) * 2.5
    data['Weighted_Seaview'] = (data['beach_proximity'] == 'Sea view').astype(int) * 2.0
    data['Weighted_Lakefront'] = (data['lake_proximity'] == 'Lakefront').astype(int) * 1.8
    data['Weighted_Lakeview'] = (data['lake_proximity'] == 'Lake view').astype(int) * 1.5
    return data

def add_location_mean_price(data: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    training_df['Price_per_cent'] = training_df['Price'] / training_df['Area']
    mean_price_per_location = (
        training_df.groupby("Location")['Price_per_cent']
        .mean()
        .rename("Mean_Price_per_Cent")
        .reset_index()
    )
    data = pd.merge(data, mean_price_per_location, on="Location", how="left")
    global_mean = training_df['Price_per_cent'].mean()
    data['Mean_Price_per_Cent'].fillna(global_mean, inplace=True)
    return data

def add_area_density_feature(data: pd.DataFrame) -> pd.DataFrame:
    data['Area_Density'] = data['Area'] * (data['density'] == 'High').astype(int)
    return data


# --- API Endpoint ---
@svc.api(input=JSON(), output=JSON())
def predict(input_data: dict) -> dict:
    
    df = pd.DataFrame([input_data])
    
    # Apply feature engineering
    df = add_weighted_features(df)
    df = add_location_mean_price(df, TRAINING_DF)
    df = add_area_density_feature(df)

    # *** THIS IS THE FIX ***
    # Select only the 8 features the model was trained on
    features_for_model = [
        'Area', 'Mean_Price_per_Cent', 'Weighted_Beachfront', 'Weighted_Seaview',
        'Weighted_Lakefront', 'Weighted_Lakeview', 'Area_Density', 'Location'
    ]
    df_for_prediction = df[features_for_model]

    # Make the prediction
    prediction = model_runner.predict.run(df_for_prediction)
    result = prediction[0]
    
    return {"predicted_price": result}