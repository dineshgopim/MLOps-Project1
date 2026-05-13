from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

class PipelineStep(ABC):        #  We make "Pipeline" class inherit from ABC to turn it into an abstract class.
    
    """
    Abstract Base Class for all steps in a machine learning pipeline.

    This class defines the "contract" that all concrete pipeline steps must follow.
    They must all implement an 'execute' method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def excute(self, data):
        """
        Executes the logic for this pipeline step.

        Args:
            data: The input data for this step (can be None, a DataFrame, etc.).

        Returns:
            The processed data after the step's logic is applied.
        """
        pass

class DataLoader(PipelineStep):     
    """ 
    class to loading the data from CSV file
    """
    
    def __init__(self, file_path: str):
        
        """
        Initializes the DataLoader with the path to the CSV file.

        Args:
            filepath (str): The path to the CSV file to load.
        """

        self.file_path = file_path
        
        super().__init__()

    def excute(self, data=None):

        """
        Loads data from the stored filepath into a pandas DataFrame.

        Args:
            data: This argument is ignored by the DataLoader but is required
                  to match the signature of the abstract method.

        Returns:
            pd.DataFrame: The loaded data.
        """

        try:
            df = pd.read_csv(self.file_path)
            print(f"Data is Loaded successfully. ")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at: {self.file_path}")
            return None

class FeatureEngineer(PipelineStep):

    """
    Concrete class for performing feature engineering on the taxi dataset.
    """
     
    def __init__(self):
        super().__init__()      # inherit from methods from parent calss

    def _haversine_distance(self, lon1, lat1, lon2, lat2):

        """
        Calculate the 'great circle distance' in kilometers between two points 
        on the earth (specified in decimal degrees).
        """

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])      # decimal to radians

        # haversine formula, which caluclates the curved distance from one point to another point on earch based on the longitude and latitude values
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371        # radius of earth
        distance = r * c
        return distance

    def excute(self, data: pd.DataFrame) -> pd.DataFrame:
      
        """
        Applies feature engineering steps to the input DataFrame.

        Args:
            data (pd.DataFrame): The raw DataFrame from the DataLoader.

        Returns:
            pd.DataFrame: The DataFrame with new and cleaned features.
        """

        df = data.copy()

        # coverting datatime
        df['pickup_datatime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

        df['hour_of_day'] = df['pickup_datatime'].dt.hour
        df['day_of_week'] = df['pickup_datatime'].dt.dayofweek
        df['month'] = df['pickup_datatime'].dt.month

        # calculating the shortest distance from one point to another based longitudes and latitudes
        df['distance_km'] = self._haversine_distance(
            df['pickup_longitude'],
            df['pickup_latitude'], 
            df['dropoff_longitude'],
            df['dropoff_latitude']
        )

        # removes unrealistic trips or cleaning outliers
        df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 7200)]
        df = df[df['distance_km'] > 0]

        print(f'Feature Engineering is complete. ')
        return df

class DataSplitter(PipelineStep):

    """
    Concrete class for splitting data into training and testing sets.
    """

    def __init__(self, features: List[str], target:str , test_size:float = 0.2, random_state: int = 42):

        """
        Initializes the DataSplitter.

        Args:
            features (List[str]): A list of column names to be used as features (X).
            target (str): The name of the target column (y).
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int): A seed for the random number generator for reproducibility.
        """
         
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        
        super().__init__()

    def excute(self, data: pd.DataFrame) -> Dict[str, Any]:
        
        """
        Splits the DataFrame into training and testing sets.

        Args:
            data (pd.DataFrame): The feature-engineered DataFrame.

        Returns:
            Dict[str, Any]: A dictionary containing X_train, X_test, y_train, y_test.
        """
       
        X = data[self.features]
        y = data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size= self.test_size, 
            random_state= self.random_state
            )
        
        print("Data Splitting is complete. ")
        return {
            "X_train": X_train,
            "X_test": X_test, 
            "y_train": y_train,
            "y_test": y_test
        }
    

class ModelTrainer(PipelineStep):

    """
    Concrete class for training a machine learning model.
    """

    def __init__(self, model):

        """
        Initializes the ModelTrainer with an untrained model object.

        Args:
            model (Any): An object with a .fit() method (e.g., a scikit-learn estimator).
        """

        self.model = model
        super().__init__()

    def excute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        
        """
        Trains the model on the provided training data.

        Args:
            data (Dict[str, Any]): A dictionary containing 'X_train' and 'y_train'.

        Returns:
            Dict[str, Any]: A dictionary containing the trained model and the test data.
        """

        X_train = data['X_train']
        y_train = data['y_train']
        
        self.model.fit(X_train, y_train)
        print(f"Model Training is Complete. ")

        return {
            'trained_model': self.model, 
            'X_test': data['X_test'],
            'y_test': data['y_test']
        }

class Pipeline:

    """
    A class to orchestrate a sequence of pipeline steps.
    """
     
    def __init__(self, steps: list[PipelineStep]):

        """
        Initializes the Pipeline with a list of steps.

        Args:
            steps (List[PipelineStep]): A list of objects that are subclasses of PipelineStep.
        """

        for step in steps:
            if not isinstance(step, PipelineStep):
                raise TypeError(f"All steps in the pipeline must be an instance of PipelineStep, but found {type(step)}")

        self.steps = steps

    def run(self) -> Any:
        
        """
        Executes all steps in the pipeline in sequence.

        The output of each step is passed as the input to the next step.

        Returns:
            Any: The final output from the last step in the pipeline.
        """

        print(f" ====== Starting Pipeline Excution ====== ")
        
        data = None

        for step in self.steps:     # The automated POLYMORPHISM, which Execute the step, passing the result of the previous step to next step
            data = step.excute(data)

        print(f" ====== Pipeline Excution Finished ====== ")
        
        return data