# Machine Learning Datasets and Models

This project provides implementations of various machine learning datasets and models, including perceptron, regression, digit classification, and language identification. in the models.py file. The code includes data handling, model training, and visualization functionalities.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages:
    ```sh
    pip install numpy matplotlib
    ```

## Usage

Run the autograder.py file to test the code. The autograder will test the perceptron, regression and digit classification.

## Dataset Classes

- `Dataset`: Base class for handling datasets.
- `PerceptronDataset`: Generates and visualizes data for a perceptron model.
- `RegressionDataset`: Generates and visualizes data for a regression model.
- `DigitClassificationDataset`: Loads and visualizes the MNIST dataset for digit classification.
- `LanguageIDDataset`: Loads and visualizes a dataset for language identification.

## Model Training

Each dataset class is designed to work with a corresponding model class. The `main` function in `backend.py` demonstrates how to initialize and train each model with its dataset.

## Visualization

The project includes visualization functionalities to help monitor the training process. These visualizations are enabled by default.

## Credits

This project was carried out as part of the INF8175 course at Polytechnique Montréal, and the files structure as well as the autograder were designed by the professor.