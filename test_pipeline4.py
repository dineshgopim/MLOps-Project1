from pipeline import DataLoader, FeatureEngineer, DataSplitter, ModelTrainer
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

print("--- Testing the Full Pipeline from Loading to Training ---")

DATA_FILEPATH = 'nyc_taxi_train.csv'

data_loader_step = DataLoader(file_path=DATA_FILEPATH)
loaded_data = data_loader_step.excute(data_loader_step)

feature_engineer_step = FeatureEngineer()
engineerd_data = feature_engineer_step.excute(loaded_data)

features_to_use = [
    'vendor_id', 
    'passenger_count', 
    'pickup_longitude', 
    'pickup_latitude',
    'dropoff_longitude', 
    'dropoff_latitude', 
    'hour_of_day', 
    'day_of_week', 
    'month', 
    'distance_km'
]

target_column = 'trip_duration'

data_splitter_step = DataSplitter(features=features_to_use, target=target_column)
split_data = data_splitter_step.excute(engineerd_data)

# model training
dt_regressor = DecisionTreeRegressor(max_depth=10, random_state=42)

training_model_step = ModelTrainer(model=dt_regressor)

training_results = training_model_step.excute(split_data)

if training_results:
    print("\n--- Verification of Model Training ---")
    trained_model = training_results['trained_model']
    X_test = training_results['X_test']
    y_test = training_results['y_test']

    print(f"Successfully returned a trained model of type: {type(trained_model)}")

    predictions = trained_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"\nModel Evaluation on Test Set:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} seconds")