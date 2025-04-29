# First import the patch
import tensorflow_patch 

# Now you can safely import the rest
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from nonlincausality import nonlincausalityNN as nlc_nn

# === PATCH: Make keras.optimizers.legacy.Adam work on Keras 3 ===
if hasattr(keras, "optimizers") and not hasattr(keras.optimizers, "legacy"):
    class LegacyOptimizers:
        Adam = keras.optimizers.Adam
    keras.optimizers.legacy = LegacyOptimizers()
# ===============================================================

class NonlinearNNGrangerCausalityAnalysis:
    def __init__(self, dataframe, max_lag=5, nn_config=['d', 'dr', 'd', 'dr'], nn_neurons=[100, 0.05, 100, 0.05]):
        if dataframe.empty or dataframe.columns.empty:
            raise ValueError("DataFrame must have at least one column representing tickers.")
        self.dataframe = dataframe
        self.tickers = dataframe.columns
        self.max_lag = max_lag
        self.nn_config = nn_config
        self.nn_neurons = nn_neurons
        self.results = {}

    def calculate_nonlinear_nn_causality(self, epochs=[50, 50], learning_rate=[0.0001, 0.00001], batch_size=32):
        for i, ticker_x in enumerate(self.tickers):
            for j, ticker_y in enumerate(self.tickers):
                if i == j:
                    self.results[(ticker_x, ticker_y)] = {
                        "p_value": np.nan,
                        "causality_score": np.nan
                    }
                else:
                    data_x = self.dataframe[ticker_x].dropna().values.reshape(-1, 1)
                    data_y = self.dataframe[ticker_y].dropna().values.reshape(-1, 1)
                    min_len = min(len(data_x), len(data_y))
                    data_x, data_y = data_x[:min_len], data_y[:min_len]
                    combined_data = np.hstack((data_x, data_y))
                    train_size = int(0.7 * len(combined_data))
                    data_train = combined_data[:train_size]
                    data_val = combined_data[train_size:]

                    result = nlc_nn(
                        x=data_train,
                        maxlag=self.max_lag,
                        NN_config=self.nn_config,
                        NN_neurons=self.nn_neurons,
                        x_test=data_val,
                        run=1,
                        epochs_num=epochs,
                        learning_rate=learning_rate,
                        batch_size_num=batch_size,
                        x_val=data_val,
                        verbose=True,
                        plot=False
                    )

                    self.results[(ticker_x, ticker_y)] = {
                        "causality_score": result['causality_score'],
                        "p_value": result['p_value']
                    }
        return self.results

    def significant_causality_pairs(self, alpha=0.05):
        significant_pairs = []
        for (ticker_x, ticker_y), result in self.results.items():
            if isinstance(result, dict) and result.get('p_value') is not None and result['p_value'] < alpha:
                significant_pairs.append((ticker_x, ticker_y))
                print(f"{ticker_x} nonlinearly causes {ticker_y} with p-value {result['p_value']}")
        return significant_pairs
