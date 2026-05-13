from pipeline import DataLoader, FeatureEngineer
import numpy as np
import pandas as pd

DATA_FILEPATH = 'nyc_taxi_train.csv'

data_loader_step = DataLoader(file_path=DATA_FILEPATH)
raw_data = data_loader_step.excute()

feature_engineering_step = FeatureEngineer()
engineered_data = feature_engineering_step.excute(raw_data)

if engineered_data is not None:
    print(f"\nShape of raw data: {raw_data.shape}")
    print(f"Shape of engineered data: {engineered_data.shape}")
    print("New columns created:")
    new_cols = set(engineered_data.columns) - set(raw_data.columns)
    print(list(new_cols))
    print("\nFirst 5 rows of engineered data:")
    print(engineered_data[['hour_of_day', 'day_of_week', 'distance_km', 'trip_duration']].head())