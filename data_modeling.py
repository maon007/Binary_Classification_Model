import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_predict, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class BinaryClassModel:

    def train_test_split_data(self, df, target_column, test_size=0.3, random_state=None):
        """
        Split the data into training and testing sets.

        Parameters:
        df (pandas.DataFrame): Input DataFrame containing features and target variable.
        target_column (str): Name of the target column.
        test_size (float or int): The proportion of the dataset to include in the test split.
        random_state (int, RandomState instance, or None): Controls the randomness of the dataset.

        Returns:
        tuple: Four DataFrames - X_train, X_test, y_train, y_test.
        """
        # Split features and target variable
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Log information about the splitting
        logging.info("Splitting the data into training and testing sets with test size: %s", test_size)

        # Perform the train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Log the sizes of the resulting sets
        logging.info("Size of X_train: %s", X_train.shape)
        logging.info("Size of X_test: %s", X_test.shape)
        logging.info("Size of y_train: %s", y_train.shape)
        logging.info("Size of y_test: %s", y_test.shape)

        return X_train, X_test, y_train, y_test

    def choose_classification_algorithm(self, X, y, cv=5, file_path='results.txt'):
        """
        Choose a suitable classification algorithm for the given task,
        store results to a text file, and return the best trained model and its evaluation metrics.

        Parameters:
          X (pandas.DataFrame): Input DataFrame containing features.
          y (pandas.Series): Target variable.
          cv (int): Number of folds in cross-validation.
          file_path (str): Path to the text file. Default is 'results.txt'.

        Returns:
          tuple: A tuple containing the best trained model (sklearn.base.BaseEstimator)
                  and a dictionary with average evaluation metrics.
        """

        best_model = None  # Store the best model here
        best_score = 0.0  # Keep track of the best score
        best_model_metrics = {}  # Store metrics for the best model

        algorithms = {
            # "Logistic Regression": LogisticRegression(),
            # "Random Forest": RandomForestClassifier(),
            "XGBoost": xgb.XGBClassifier(n_estimators=500, min_child_weight=9, max_depth=8, gamma=0, learning_rate=0.1),
            # "XGBoost": xgb.XGBClassifier(),
            # "Support Vector Machine": SVC(),
        }

        with open(file_path, 'a') as file:
            for name, model in algorithms.items():
                # Perform cross-validation using cross_val_score
                accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision_macro')
                recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall_macro')
                f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

                # Calculate average metrics
                average_accuracy = accuracy_scores.mean()
                average_precision = precision_scores.mean()
                average_recall = recall_scores.mean()
                average_f1 = f1_scores.mean()

                # Write evaluation metrics and algorithm name to the text file
                file.write(f"Algorithm: {name}\n")
                file.write(f"Average Accuracy: {average_accuracy:.4f}\n")
                file.write(f"Average Precision: {average_precision:.4f}\n")
                file.write(f"Average Recall: {average_recall:.4f}\n")
                file.write(f"Average F1-score: {average_f1:.4f}\n")
                file.write("-" * 50 + "\n")

                # Print evaluation metrics
                logging.info("Evaluation metrics for %s:", name)
                logging.info("Average Accuracy: %.4f", average_accuracy)
                logging.info("Average Precision: %.4f", average_precision)
                logging.info("Average Recall: %.4f", average_recall)
                logging.info("Average F1-score: %.4f", average_f1)
                logging.info("-" * 50)

                # Update best model and metrics
                if average_f1 > best_score:
                    best_model = model
                    best_score = average_f1
                    best_model_metrics = {
                        'accuracy': average_accuracy,
                        'precision': average_precision,
                        'recall': average_recall,
                        'f1': average_f1
                    }

        # Fit the best model on the entire dataset
        best_model.fit(X, y)

        return best_model

    def evaluate_on_test_data(self, model, X_test, y_test, visualize_confusion_matrix=False):
        """
        Make predictions on the test data using the provided model and evaluate its performance.

        Parameters:
            model: Trained classification model.
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): True labels for the test data.
            visualize_confusion_matrix (bool, optional): Flag to control confusion matrix visualization. Defaults to False.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print evaluation metrics
        print("Evaluation metrics on the test set:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Visualize confusion matrix if requested
        if visualize_confusion_matrix:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.show()

        # Return evaluation metrics as a dictionary
        evaluation_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "confusion_matrix": conf_matrix
        }
        return evaluation_metrics
