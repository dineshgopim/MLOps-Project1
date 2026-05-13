# Object-Oriented ML Pipeline Framework

## Project Overview

This project is a custom, object-oriented machine learning pipeline framework built from scratch in Python. 
It demonstrates core software engineering principles (OOP) applied to a real-world machine learning problem: predicting taxi trip durations in New York City.

The framework is designed to be modular, reusable, and easy to understand, where each stage of the ML workflow is encapsulated into its own class, all inheriting from a common abstract base class.

**Dataset:** The project uses the "New York City Taxi Trip Duration" dataset from a Kaggle competition, which contains over 1.4 million trip records.

---

## Key Features & OOP Concepts Demonstrated

This framework was built as a capstone project to demonstrate a deep understanding of Object-Oriented Programming principles:

*   **Abstraction:** An abstract base class, `PipelineStep`, defines a "contract" that all pipeline components must follow. It enforces the implementation of an `.execute()` method, ensuring a consistent interface across the framework.

*   **Inheritance:** Concrete classes (`DataLoader`, `FeatureEngineer`, `DataSplitter`, `ModelTrainer`) inherit from the `PipelineStep` ABC, reusing its structure while providing their own specific implementations.

*   **Polymorphism:** The main `Pipeline` orchestrator class iterates through a list of `PipelineStep` objects and calls the same `.execute()` method on each one. Each object responds in its own unique, polymorphic way (loading data, engineering features, or training a model).

*   **Encapsulation:** Each class encapsulates its own logic and data. For example, the `DataLoader` encapsulates the file path, and the `ModelTrainer` encapsulates the model object it is responsible for training.

---

## How to Run the Project

This project was developed in a Python 3.10 environment.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd ML-Pipeline-Framework
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    The project requires the Kaggle API for data download. Ensure you have your `kaggle.json` token set up.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the data:**
    ```bash
    kaggle competitions download -c nyc-taxi-trip-duration
    unzip nyc-taxi-trip-duration.zip
    unzip train.zip
    mv train.csv nyc_taxi_train.csv
    ```

5.  **Run the pipeline:**
    ```bash
    python main.py
    ```

---

## Final Result

The pipeline trains a `DecisionTreeRegressor` model and evaluates it using Root Mean Squared Error (RMSE).

*   **Final Model RMSE:** Approximately 335.75 seconds (~5.6 minutes).