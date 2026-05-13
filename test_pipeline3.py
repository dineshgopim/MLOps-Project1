from pipeline import DataLoader, FeatureEngineer, DataSplitter

DATA_FILEPATH = 'nyc_taxi_train.csv'

data_loader_step = DataLoader(file_path=DATA_FILEPATH)
raw_data = data_loader_step.excute()

feature_engineering_step = FeatureEngineer()
engineered_data = feature_engineering_step.excute(raw_data)

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
split_data = data_splitter_step.excute(engineered_data)


if split_data:
    print("\n--- Verification of Data Splitting ---")
    print(f"Type of split_data: {type(split_data)}")
    print(f"Keys in the dictionary: {split_data.keys()}")
    
    # Print the shapes of the resulting datasets
    print(f"Shape of X_train: {split_data['X_train'].shape}")
    print(f"Shape of X_test: {split_data['X_test'].shape}")
    print(f"Shape of y_train: {split_data['y_train'].shape}")
    print(f"Shape of y_test: {split_data['y_test'].shape}")