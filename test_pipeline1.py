from pipeline import DataLoader, FeatureEngineer
import pandas as pd

print(f"--- Testing The Dataset Loader ---")

DATA_FILEPATH = 'nyc_taxi_train.csv'

data_loader_step = DataLoader(file_path=DATA_FILEPATH)
loaded_data = data_loader_step.excute()

if loaded_data is not None:
    print(f"Successfully returned an object of type: {type(loaded_data)}")
    print(f"Shape of the loaded data: {loaded_data.shape}")
    print("First 5 rows of the data:")
    print(loaded_data.head())


