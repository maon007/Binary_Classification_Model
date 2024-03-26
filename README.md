# Binary Classification Model

## Overview

This Python code provides a comprehensive framework for building, evaluating, and tuning binary classification models using various machine learning algorithms. It includes functionalities such as data loading, preprocessing, feature engineering, model selection, and evaluation.

## Features

- **Data Loading**: Load CSV data and convert the timestamp column to the appropriate format.
- **Data Exploration**: Visualize the distribution of the target variable and perform basic statistical analysis.
- **Data Preprocessing**: Handle missing values, duplicate rows, outliers, encode categorical features, scale numerical features, and extract datetime features.
- **Model Training**: Choose the best classification algorithm using cross-validation and evaluate its performance.
- **Model Evaluation**: Calculate evaluation metrics including accuracy, precision, recall, and F1-score on the test set.
- **Logging**: Utilize logging to track the execution flow and provide information about data processing and model evaluation.

## Usage

### Direct way

1. Clone the repository:
   ```bash
   git clone https://github.com/maon007/Binary_Classification_Model.git
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```

#### Running the Classification Model
To analyze the data, build, evaluate and tune the model follow these steps:
- Ensure you have a CSV file containing the data (e.g., it_data.csv).
- Run the main Python script with the path to the CSV file as an argument (e.g.):
   ```bash
    python main.py "./it_data.csv"
   ```
- The scripts will load the data, run the general exploration and analysis of the data, build, evaluate, and tune the classification model


### Using Dockerfiles
You can also use a Dockerfile to build the image for the scripts. Open a terminal and navigate to the directory containing Dockerfiles and Python files. Build the image using the following command:
```bash
docker build -t my_docker_image -f Dockerfile .
```
You can replace "my_docker_image" with the desired name for your Docker image.

Run the Docker container: Once the image is built, you can run it as a container:
```bash
docker run my_docker_image
```


**NOTE:** Docker Image was tested and run successfully.
