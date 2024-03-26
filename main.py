import logging
import argparse
from data_exploration import DataExploration
from data_processing import DataProceesing
from data_modeling import BinaryClassModel


def main(csv_file):
    # Set up logging configuration
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create an instance of the DataExploration class with the CSV file path
    data_exploration = DataExploration('it_data.csv')
    # Load CSV data and log the dimensions of the DataFrame
    df = data_exploration.load_data()
    data_exploration.visualize_target_distribution(df=df, target_column="TARGET", visualize_target=False)
    data_exploration.statistics(df=df, describe=False, unique_count=False, missing_count=False, data_types=False)
    data_exploration.count_columns_by_datatype(df=df)
    df = data_exploration.find_duplicate_rows(df=df)
    data_exploration.null_percentage(df=df, just_nulls=True)
    data_exploration.count_rows_with_null(df=df)
    data_processing = DataProceesing()
    df = data_processing.drop_missing_rows(df=df)
    # model.correlation_analysis(df=df)
    df = data_processing.detect_and_remove_outliers(df=df, z_thresh=3, visualize=False, apply=True)
    df = data_processing.extract_datetime_features(df=df, enable=True)
    df = data_processing.encode_categorical_features(df=df, enc="one_hot")
    # df = data_processing.scale_features(df=df, scaling="no_scaling")
    df = data_processing.select_relevant_features(df=df)
    # data_exploration.correlation_analysis(df=df)
    # data_exploration.visualize_target_distribution(df=df, target_column="TARGET", visualize_target=True)
    data_modeling = BinaryClassModel()
    X_train, X_test, y_train, y_test = data_modeling.train_test_split_data(df=df, target_column="TARGET")
    # Tune hyperparameters
    # best_params, best_model = data_modeling.tune_hyperparameters(X_train, y_train)
    # Call the choose_classification_algorithm method to train the model
    cls = data_modeling.choose_classification_algorithm(X=X_train, y=y_train, cv=5)
    # Make predictions on the test data
    data_modeling.evaluate_on_test_data(model=cls, X_test=X_test, y_test=y_test, visualize_confusion_matrix=False)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Train a binary classification model using the provided CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the data.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with the provided CSV file path
    main(args.csv_file)
