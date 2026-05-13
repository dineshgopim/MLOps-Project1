from pipeline import DataLoader, FeatureEngineer, DataSplitter, ModelTrainer, Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

DATA_FILEPATH = 'nyc_taxi_train.csv'

FEATURES_TO_USE = [
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

TARGET_COLUMN = 'trip_duration'

def main():

    """
    The main function to define and run the ML pipeline.
    """

    pipeline_steps = [
        DataLoader(file_path=DATA_FILEPATH),
        FeatureEngineer(),
        DataSplitter(features=FEATURES_TO_USE, target=TARGET_COLUMN),
        ModelTrainer(model=DecisionTreeRegressor(max_depth=10, random_state=42))
    ]

    ml_pipeline = Pipeline(steps=pipeline_steps)

    final_results = ml_pipeline.run()

    if final_results:

        print("\n--- Final Model Evaluation ---")
        trained_model = final_results['trained_model']
        X_test = final_results['X_test']
        y_test = final_results['y_test']
        
        predictions = trained_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Final Model RMSE on Test Set: {rmse:.2f} seconds")

# This standard Python construct ensures that main() is called when you run 'python main.py'
if __name__ == "__main__":
    main()

