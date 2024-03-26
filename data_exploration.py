import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class DataExploration:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Load CSV data from the specified file path and convert the 'time'
        column to timestamp format. Additionally, check the distribution of
        values for the target variable 'TARGET' and print it via logging.

        Returns:
            pandas.DataFrame: DataFrame containing the loaded data.
        """
        logging.info("Loading data from file: %s", self.file_path)
        # low_memory - handling mixed data type columns
        data = pd.read_csv(self.file_path, low_memory=False)

        # Convert timeslot_datetime_from from object to timestamp format
        data['TIME'] = pd.to_datetime(data['TIME'])

        data = data.sort_values(by=['TIME'])
        logging.info("Data dimensions after loading: %s", data.shape)

        return data

    def visualize_target_distribution(self, df, target_column, visualize_target=True):
        """
        Count and visualize the distribution of values for the target variable.

        Parameters:
            df (pandas.DataFrame): DataFrame containing the data.
            target_column (str): Name of the target column.
            visualize_target (bool): Whether to visualize the distribution. Default is True.

        Returns:
            None
        """
        # Count the occurrences of each value in the target variable
        target_counts = df[target_column].value_counts()

        # Calculate percentages
        total_count = len(df)
        target_percentages = (target_counts / total_count) * 100

        if visualize_target:
            # Visualize the distribution using a bar plot
            plt.figure(figsize=(8, 6))
            sns.countplot(data=df, x=target_column)
            plt.title(f'Distribution of {target_column}')
            plt.xlabel(target_column)
            plt.ylabel('Count')
            plt.show()

        # Print the count and percentage of each value in the target variable
        logging.info("Distribution of values for the target variable:")
        for value, count, percentage in zip(target_counts.index, target_counts, target_percentages):
            logging.info("%s: Count=%d, Percentage=%.2f%%", value, count, percentage)

    def statistics(self, df, describe=True, unique_count=False, missing_count=False, data_types=False):
        """
        Display statistics for a pandas DataFrame including count of unique values, count of missing values,
        and data types of columns.

        Parameters:
        - df (DataFrame): The pandas DataFrame for which extended statistics are to be calculated.
        """
        try:
            if describe:
                # Basic statistics for numerical columns
                print("Basic statistics for numerical columns:")
                print(df.describe())

            if unique_count:
                # Count of unique values for each column
                print("\nCount of unique values for each column:")
                print(df.nunique())

            if missing_count:
                # Count of missing values for each column
                print("\nCount of missing values for each column:")
                print(df.isnull().sum())

            if data_types:
                # Data types of columns
                print("\nData types of columns:")
                print(df.dtypes)

        except Exception as e:
            print("An error occurred:", e)

    def count_columns_by_datatype(self, df):
        """
        Count the number of columns for each data type in a DataFrame.

        Parameters:
        df (pandas.DataFrame): Input DataFrame.

        Returns:
        dict: Dictionary containing the count of columns for each data type.
        """
        # Get the data types of columns
        dtypes = df.dtypes
        count_by_datatype = dtypes.value_counts().to_dict()

        logging.info("Count of columns for each data type: %s", count_by_datatype)

        return count_by_datatype

    def find_duplicate_rows(self, df):
        """
        Find duplicate rows in a pandas DataFrame.

        Parameters:
        - df (DataFrame): The pandas DataFrame to be checked for duplicate rows.

        Returns:
        - DataFrame: The DataFrame with duplicate rows removed.
        """
        try:
            # Check for duplicate rows
            duplicate_rows = df[df.duplicated()]

            # If duplicate rows are found
            if not duplicate_rows.empty:
                num_duplicates = len(duplicate_rows)
                total_rows = len(df)
                duplicate_percentage = (num_duplicates / total_rows) * 100

                logging.info("Number of duplicate rows found:", num_duplicates)
                logging.info("Percentage of duplicate rows from the total:", duplicate_percentage, "%")

                return df
            else:
                logging.info("No duplicate rows found.")
                return df
        except Exception as e:
            print("An error occurred:", e)
            return None

    def null_percentage(self, df, just_nulls=False):
        """
        Calculate and display the percentage of null values for each column in a pandas DataFrame.

        Parameters:
        - df (DataFrame): The pandas DataFrame for which null percentages are to be calculated.
        - just_nulls (bool): If True, show only columns with all null values.
        """
        try:
            # Calculate the percentage of null values for each column
            null_percentages = (df.isnull().sum() / len(df)) * 100

            if just_nulls:
                # Filter columns with all null values
                null_columns = null_percentages[null_percentages == 100]
                if not null_columns.empty:
                    logging.info("Columns with all null values:")
                    logging.info(null_columns)
                else:
                    logging.info("No columns with all null values found.")
            else:
                # Display the percentage of null values for each column
                logging.info("Percentage of null values for each column:")
                logging.info(null_percentages)
        except Exception as e:
            logging.error("An error occurred: %s - %s", type(e).__name__, e)

    def count_rows_with_null(self, df):
        """
        Calculate the number of rows containing null values in a pandas DataFrame.

        Parameters:
        - df (DataFrame): The pandas DataFrame to be checked for null values.

        Returns:
        - int: The number of rows containing null values.
        """
        logging.info("Total number of rows: %d", df.shape[0])
        # Calculate the number of rows containing null values
        rows_with_null = df.isnull().any(axis=1).sum()
        logging.info("Number of rows containing null values: %d", rows_with_null)
        return rows_with_null

    def correlation_analysis(self, df):
        """
        Perform correlation analysis for numerical features in a DataFrame.
        Highly correlated features may not provide additional information to the model

        Parameters:
            df (pandas.DataFrame): Input DataFrame containing numerical features.

        Returns:
            pandas.DataFrame: DataFrame containing the correlation matrix.
        """
        # Select only numerical columns
        numerical_columns = df.select_dtypes(include=[np.number])

        # Calculate the correlation matrix
        correlation_matrix = numerical_columns.corr()

        # Visualize the correlation matrix as a heatmap
        plt.figure(figsize=(17, 17))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

        return correlation_matrix