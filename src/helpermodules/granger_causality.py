import pandas as pd
import numpy as np
from helpermodules.memory_handling import PickleHelper
from statsmodels.tsa.stattools import grangercausalitytests

class GrangerCausalityAnalysis:
    """
    A class to perform Granger causality analysis between pairs of stock tickers.
    
    Attributes:
        dataframe (pandas.DataFrame): The DataFrame containing stock prices.
        max_lag (int): The maximum lag to test for Granger causality.ß
    """
    
    def __init__(self, dataframe, max_lag=5):
        """
        Initialize the GrangerCausalityAnalysis object.
        
        Args:
            dataframe (pandas.DataFrame): DataFrame containing stock prices.
            max_lag (int): Maximum lag to test for Granger causality.
        """
        # Validate that the DataFrame has columns
        if dataframe.empty or dataframe.columns.empty:
            raise ValueError("DataFrame must have at least one column representing tickers.")
        self.dataframe = dataframe
        self.tickers = dataframe.columns  # Automatically get tickers from DataFrame columns
        
        self.max_lag = max_lag
        self.results = {}  # To store Granger causality results for each pair

    def calculate_granger_causality(self):
        """
        Calculate Granger causality for each pair of tickers.
        
        Returns:
            dict: A dictionary containing Granger causality test results for each ticker pair.
        """
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                ticker_x = self.tickers[i]
                ticker_y = self.tickers[j]
                    
                if i == j:
                    # If testing the same ticker, set NaNs
                    self.results[(ticker_x, ticker_y)] = {
                    "p_values": [np.nan] * self.max_lag,
                    "f_statistics": [np.nan] * self.max_lag
                    }
                else:
                    # Prepare data for Granger causality test by filling NaNs
                    data = self.dataframe[[ticker_x, ticker_y]].fillna(method='ffill').fillna(method='bfill')
                    # Run Granger causality test
                    result = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
                    # Extract and store p-values and F-statistics for each lag
                    p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, self.max_lag + 1)]
                    f_stats = [result[lag][0]['ssr_ftest'][0] for lag in range(1, self.max_lag + 1)]
                    # Store results in a dictionary
                    self.results[(ticker_x, ticker_y)] = {
                        "p_values": p_values,
                        "f_statistics": f_stats
                    }
        return self.results
    

    def significant_causality_pairs(self, alpha=0.05):
        """
        Identify ticker pairs with significant Granger causality.
        
        Args:
            alpha (float): Significance level threshold for p-values (default is 0.05).
        
        Returns:
            list: List of ticker pairs with Granger causality (based on p-value threshold).
        """
        significant_pairs = []
        
        for (ticker_x, ticker_y), result in self.results.items():
            # Check if any p-value is below the significance level
            if any(p < alpha for p in result['p_values']):
                significant_pairs.append((ticker_x, ticker_y))
                print(f"{ticker_x} Granger-causes {ticker_y} with p-values {result['p_values']}")
        PickleHelper(significant_pairs).pickle_dump('Linear_Granger_Casuality_Significants')
        return significant_pairs
