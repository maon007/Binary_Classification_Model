import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import TargetEncoder, StandardScaler, MinMaxScaler


class DataProceesing:

    def drop_missing_rows(self, df):
        """
        Drop rows with missing values in a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): Input DataFrame containing missing values.

        Returns:
        pandas.DataFrame: DataFrame with missing rows dropped.
        """
        # Drop rows with any missing values
        df.dropna(axis=0, how='any', inplace=True)

        # Log the shape of the DataFrame after dropping rows
        logging.info("Data dimensions after dropping rows containing NULL values: %s", df.shape)

        return df

    def detect_and_remove_outliers(self, df, z_thresh=3, visualize=False, apply=False):
        """
        Detect outliers in float numerical columns of the dataset using the Z-score method.

        Parameters:
            df (pandas.DataFrame): Input DataFrame.
            threshold (float): Threshold value for outlier detection. Default is 3.
            visualize (bool): Whether to visualize the distribution of data for each column. Default is False.
            apply (bool): Whether to apply outlier removal. Default is False.

        Returns:
            pandas.DataFrame: DataFrame containing outlier information for each column.
        """

        def is_outlier(x):
            """
            Returns True if a value is an outlier based on the Z-score threshold.
            """
            if not pd.api.types.is_numeric_dtype(x):
                return False
            return abs(x - x.mean()) > z_thresh * x.std()

        # Select float columns and detect outliers
        float_cols = df.select_dtypes(include=[np.float64])
        outlier_indices = []
        for col in float_cols:
            outlier_mask = df[col].apply(is_outlier)
            outlier_indices.extend(df[outlier_mask].index.tolist())

        # Combine outlier indices and remove duplicates
        outlier_indices = list(set(outlier_indices))

        # Log the number of total outliers detected
        logging.info(f"Total number of outliers detected: {len(outlier_indices)}")

        # Visualize box plots if visualize is True
        if visualize:
            num_plots = len(float_cols)
            num_cols = min(num_plots, 5)
            num_rows = (num_plots + num_cols - 1) // num_cols

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
            axes = axes.flatten()

            for i, col in enumerate(float_cols):
                sns.boxplot(x=df[col], ax=axes[i])
                axes[i].set_title(f'Box plot of {col}')
                axes[i].set_xlabel(col)

            plt.tight_layout()
            plt.show()

        # Apply outlier removal if apply is True
        if apply:
            logging.info("Applying outlier removal...")
            df = df.drop(outlier_indices)
            logging.info("Outlier removal applied. DataFrame dimension after removal: %s", df.shape)
        else:
            logging.info("Applying outlier removal is skipped.")

        return df

    def extract_datetime_features(self, df, enable=False):
        """
        Extract datetime features from the timeslot_datetime_from column.

        Parameters:
            data (DataFrame): DataFrame containing the CSV data with 'timeslot_datetime_from' column.
            enable (bool): Flag to enable/disable datetime feature extraction. Default is True.

        Returns:
            DataFrame: DataFrame with extracted datetime features.
        """
        if not enable:
            logging.info("Datetime feature extraction is disabled. Skipping...")
            return df

        else:
            logging.info("Generating new features: day_of_week, month_of_year and season")

            df['day_of_week'] = df['TIME'].dt.day_name()
            df['month_of_year'] = df['TIME'].dt.month_name()
            # df['hour_of_day'] = df['timeslot_datetime_from'].dt.hour

            # Define seasons based on months
            month_to_season = {
                1: 'Winter', 2: 'Winter', 3: 'Spring',
                4: 'Spring', 5: 'Spring', 6: 'Summer',
                7: 'Summer', 8: 'Summer', 9: 'Fall',
                10: 'Fall', 11: 'Fall', 12: 'Winter'
            }
            df['season'] = df['TIME'].dt.month.map(month_to_season)

            return df

    def encode_categorical_features(self, df, enc="one_hot"):
        """
        Encode categorical features using either one-hot encoding or target encoding.
        Target encoding captures the relationship between the categorical variable and the target variable,
            - potentially improving the predictive performance of the model.

        Parameters:
            data (DataFrame): DataFrame containing the CSV data.
            enc (str): Encoding method ("one-hot" or "target"). Default is "one-hot".

        Returns:
            DataFrame: DataFrame with categorical features encoded.
        """

        # Get list of object columns (categorical features)
        cat_columns = df.select_dtypes(include=['object', 'bool']).columns

        if enc == "one_hot":
            # Perform one-hot encoding
            data_encoded = pd.get_dummies(df, columns=cat_columns, dtype=int)
            logging.info("Data dimensions after one-hot encoding: %s", data_encoded.shape)
        elif enc == "target":
            # Implement target encoding
            encoder = TargetEncoder()
            data_encoded = df.copy()
            data_encoded[cat_columns] = encoder.fit_transform(df[cat_columns], df["TARGET"])
            logging.info("Data dimensions after target encoding: %s", data_encoded.shape)
        else:
            raise ValueError(f"Invalid encoding method '{enc}'. Use 'one-hot' or 'target'.")

        return data_encoded

    def scale_features(self, df, scaling="standard"):
        """
        This function applies different scaling methods to a pandas DataFrame based on the input parameter.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            scaling (str, optional): The type of scaling to apply.
                Can be "standard" (default), "minmax", or "no_scaling".

        Returns:
            pandas.DataFrame: The DataFrame with scaled or original columns.
        """

        float_cols = df.select_dtypes(include=['float64']).columns
        if scaling == "standard":
            scaler = StandardScaler()
        elif scaling == "minmax":
            scaler = MinMaxScaler()
        elif scaling == "no_scaling":
            logging.info("No scaling applied.")
            return df  # Skip scaling
        else:
            raise ValueError("Invalid scaling parameter. Choose 'standard', 'minmax', or 'no_scaling'.")

        if scaling != "no_scaling":
            scaler.fit(df[float_cols])
            df[float_cols] = scaler.transform(df[float_cols])
            logging.info("Data dimensions after standard scaling: %s", df.shape)
        return df

    def select_relevant_features(self, df):
        """
        Select relevant features in a pandas DataFrame by excluding specific columns.

        Parameters:
        df (pandas.DataFrame): Input DataFrame containing features.

        Returns:
        pandas.DataFrame: DataFrame with irrelevant columns excluded.
        """
        # List of columns to exclude
        exclude_columns = ['ID_APPLICATION', 'TIME']

        # Check if the columns to exclude are present in the DataFrame
        for column in exclude_columns:
            if column not in df.columns:
                logging.warning("Column '%s' not found in the DataFrame.", column)

        # Drop the specified columns
        df = df.drop(columns=exclude_columns, errors='ignore')

        # Log information about excluded columns
        logging.info("Irrelevant features excluded from the DataFrame.")

        return df
