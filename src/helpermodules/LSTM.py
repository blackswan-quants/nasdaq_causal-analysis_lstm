import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class LSTMRegressor:
    """
    A class for building, training, and evaluating an LSTM-based regression model.

    Attributes:
        feature_columns (list): List of feature column names.
        target_column (str): Name of the target column.
        input_sequence_length (int): Length of input sequences for the LSTM model.
        batch_size (int): Batch size for training and testing.
        split_fraction (float): Fraction of data to use for training (remainder is for testing).
        scaler_features (StandardScaler): Scaler for normalizing feature data.
        scaler_target (StandardScaler): Scaler for normalizing target data.
        model (Sequential): The trained LSTM model.
    """

    def __init__(self, feature_columns, target_column, input_sequence_length=60, batch_size=32, split_fraction=0.7):
        """
        Initializes the LSTMRegressor with the specified parameters.

        Args:
            feature_columns (list): List of feature column names.
            target_column (str): Name of the target column.
            input_sequence_length (int, optional): Length of input sequences. Defaults to 60.
            batch_size (int, optional): Batch size for training and testing. Defaults to 32.
            split_fraction (float, optional): Fraction of data to use for training. Defaults to 0.7.
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.input_sequence_length = input_sequence_length
        self.batch_size = batch_size
        self.split_fraction = split_fraction
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.model = None

    def prepare_data(self, df):
        """
        Prepares the data for training and testing by normalizing and splitting it.

        Args:
            df (pd.DataFrame): The input DataFrame containing feature and target columns.

        Returns:
            tuple: A tuple containing the training and testing datasets.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        # Validate columns
        missing = [col for col in self.feature_columns + [self.target_column] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        features = df[self.feature_columns].values
        target = df[[self.target_column]].values

        # Normalize
        normalized_features = self.scaler_features.fit_transform(features)
        normalized_target = self.scaler_target.fit_transform(target)

        # Split
        split_index = int(len(df) * self.split_fraction)
        self.train_dataset = TimeseriesGenerator(
            data=normalized_features[:split_index],
            targets=normalized_target[:split_index],
            length=self.input_sequence_length,
            batch_size=self.batch_size
        )
        self.test_dataset = TimeseriesGenerator(
            data=normalized_features[split_index:],
            targets=normalized_target[split_index:],
            length=self.input_sequence_length,
            batch_size=self.batch_size
        )
        return self.train_dataset, self.test_dataset

    def build_and_train_model(self, epochs=20, lr=0.0001):
        """
        Builds and trains the LSTM model.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 20.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.0001.

        Raises:
            RuntimeError: If prepare_data() has not been called before training.
        """
        if not hasattr(self, 'train_dataset'):
            raise RuntimeError("You must call prepare_data() before training the model.")

        input_shape = self.train_dataset[0][0].shape[1:]

        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

        model.fit(self.train_dataset, epochs=epochs, callbacks=[early_stopping], verbose=1)
        self.model = model

    def predict(self):
        """
        Generates predictions using the trained model.

        Returns:
            tuple: A tuple containing the actual and predicted values in their original scales.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        predictions, actuals = [], []
        for x_batch, y_batch in self.test_dataset:
            pred_batch = self.model.predict(x_batch, verbose=0)
            predictions.append(pred_batch)
            actuals.append(y_batch)

        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)

        predictions_original = self.scaler_target.inverse_transform(predictions)
        actuals_original = self.scaler_target.inverse_transform(actuals)

        return actuals_original, predictions_original

    def evaluate_and_plot(self):
        """
        Evaluates the model's performance and plots the actual vs predicted values.

        Prints:
            Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-Squared (R²) scores.

        Plots:
            A line plot comparing actual and predicted values.
        """
        actuals_original, predictions_original = self.predict()

        mse = mean_squared_error(actuals_original, predictions_original)
        mae = mean_absolute_error(actuals_original, predictions_original)
        r2 = r2_score(actuals_original, predictions_original)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-Squared (R²): {r2:.4f}")

        plt.figure(figsize=(12, 6))
        plt.plot(actuals_original, label='Actual', color='blue')
        plt.plot(predictions_original, label='Predicted', color='orange', linestyle='dashed')
        plt.title("Actual vs Predicted")
        plt.xlabel("Time Step")
        plt.ylabel(self.target_column)
        plt.legend()
        plt.tight_layout()
        plt.show()
