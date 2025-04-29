# statistical_analysis.py

"""
This module provides statistical analysis tools for stock data, including functions for outlier detection and visualization.

Classes:
- StatisticalAnalysis: Provides methods for computing statistical measures and detecting outliers in stock data.

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- scipy.stats
- helpermodules.memory_handling.PickleHelper

Usage:
- Initialize the StatisticalAnalysis class with a pandas DataFrame containing stock prices.
- Use methods to compute statistical measures, detect outliers, and visualize data.

Example:
    from helpermodules.statistical_analysis import StatisticalAnalysis
    from helpermodules.memory_handling import PickleHelper
    import pandas as pd

    # Load your stock data into a DataFrame
    df = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)

    # Initialize the StatisticalAnalysis object
    analysis = StatisticalAnalysis(df)

    # Compute basic statistics
    stats = analysis.compute_statistics()

    # Detect outliers using rolling window z-score method
    outliers = analysis.detect_outliers(method='rolling_z_score', window=20, threshold=2)

    # Visualize the outliers
    analysis.plot_outliers(outliers)

    # Save the results
    PickleHelper(stats).pickle_dump('basic_statistics.pkl')
    PickleHelper(outliers).pickle_dump('detected_outliers.pkl')
"""

# Libraries used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from helpermodules.memory_handling import PickleHelper

class StatisticalAnalysis:
    """
    A class for performing statistical analysis on stock data, including outlier detection and visualization.

    Attributes:
        dataframe (pandas.DataFrame): The DataFrame containing stock data, with dates as index and tickers as columns.

    Methods:
        compute_statistics(): Compute basic statistical measures (mean, median, std, etc.) for each stock.
        detect_outliers(method='z_score', **kwargs): Detect outliers in the stock data using the specified method.
        plot_outliers(outliers): Visualize the detected outliers.
        plot_time_series(ticker): Plot the time series of a specific ticker.
        plot_distribution(ticker): Plot the distribution (histogram) of a specific ticker's returns.
    """

    def __init__(self, dataframe):
        """
        Initialize the StatisticalAnalysis object.

        Args:
            dataframe (pandas.DataFrame): DataFrame containing stock prices, with dates as index and tickers as columns.
        """
        self.dataframe = dataframe

    def compute_statistics(self):
        """
        Compute basic statistical measures (mean, median, standard deviation, skewness, kurtosis) for each stock.

        Returns:
            pandas.DataFrame: A DataFrame containing the statistical measures for each stock.
        """
        stats_df = pd.DataFrame(index=self.dataframe.columns)
        stats_df['Mean'] = self.dataframe.mean()
        stats_df['Median'] = self.dataframe.median()
        stats_df['Std'] = self.dataframe.std()
        stats_df['Variance'] = self.dataframe.var()
        stats_df['Skewness'] = self.dataframe.skew()
        stats_df['Kurtosis'] = self.dataframe.kurtosis()

        return stats_df

    def detect_outliers(self, method='z_score', **kwargs):
        """
        Detect outliers in the stock data using the specified method.

        Args:
            method (str): The method to use for outlier detection. Options are:
                - 'z_score': Standard Z-score method.
                - 'modified_z_score': Modified Z-score using median and MAD.
                - 'iqr': Interquartile Range method.
                - 'rolling_z_score': Rolling window Z-score method.
                - 'rolling_quantile': Rolling window quantile method.
            **kwargs: Additional keyword arguments specific to each method.

        Returns:
            pandas.DataFrame: A DataFrame indicating the presence of outliers for each stock at each time point.
        """
        if method == 'z_score':
            threshold = kwargs.get('threshold', 3)
            z_scores = np.abs(stats.zscore(self.dataframe, nan_policy='omit'))
            outliers = (z_scores > threshold)
        elif method == 'modified_z_score':
            threshold = kwargs.get('threshold', 3.5)
            median = self.dataframe.median()
            mad = self.dataframe.mad()
            modified_z_scores = 0.6745 * (self.dataframe - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
        elif method == 'iqr':
            threshold = kwargs.get('threshold', 1.5)
            Q1 = self.dataframe.quantile(0.25)
            Q3 = self.dataframe.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.dataframe < (Q1 - threshold * IQR)) | (self.dataframe > (Q3 + threshold * IQR)))
        elif method == 'rolling_z_score':
            window = kwargs.get('window', 20)
            threshold = kwargs.get('threshold', 3)
            rolling_mean = self.dataframe.rolling(window=window, min_periods=1).mean()
            rolling_std = self.dataframe.rolling(window=window, min_periods=1).std()
            z_scores = (self.dataframe - rolling_mean) / rolling_std
            outliers = np.abs(z_scores) > threshold
        elif method == 'rolling_quantile':
            window = kwargs.get('window', 20)
            lower_quantile = kwargs.get('lower_quantile', 0.05)
            upper_quantile = kwargs.get('upper_quantile', 0.95)
            rolling_quantiles = self.dataframe.rolling(window=window, min_periods=1).quantile([lower_quantile, upper_quantile]).transpose(2, 0, 1)
            lower_bound = rolling_quantiles[:, :, 0]
            upper_bound = rolling_quantiles[:, :, 1]
            outliers = ((self.dataframe < lower_bound) | (self.dataframe > upper_bound))
        else:
            raise ValueError("Method must be one of 'z_score', 'modified_z_score', 'iqr', 'rolling_z_score', or 'rolling_quantile'.")

        return outliers

    def plot_outliers(self, outliers):
        """
        Visualize the detected outliers.

        Args:
            outliers (pandas.DataFrame): A DataFrame indicating the presence of outliers for each stock at each time point.

        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(outliers.T, cmap='Reds', cbar=False)
        plt.xlabel('Date')
        plt.ylabel('Ticker')
        plt.title('Outlier Detection Heatmap')
        plt.show()

    def plot_time_series(self, ticker, with_outliers=False, outliers=None):
        """
        Plot the time series of a specific ticker.

        Args:
            ticker (str): The ticker symbol of the stock to plot.
            with_outliers (bool): Whether to highlight outliers on the plot.
            outliers (pandas.DataFrame): DataFrame indicating outliers, required if with_outliers is True.

        Returns:
            None
        """
        if ticker not in self.dataframe.columns:
            raise ValueError(f"Ticker '{ticker}' not found in DataFrame columns.")

        plt.figure(figsize=(12, 6))
        plt.plot(self.dataframe.index, self.dataframe[ticker], label=ticker)

        if with_outliers:
            if outliers is None:
                raise ValueError("Outliers DataFrame must be provided when with_outliers is True.")
            outlier_dates = self.dataframe.index[outliers[ticker]]
            outlier_values = self.dataframe[ticker][outliers[ticker]]
            plt.scatter(outlier_dates, outlier_values, color='red', label='Outliers')

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Time Series of {ticker}')
        plt.legend()
        plt.show()

    def plot_distribution(self, ticker, bins=50):
        """
        Plot the distribution (histogram) of a specific ticker's returns.

        Args:
            ticker (str): The ticker symbol of the stock to plot.
            bins (int): Number of histogram bins.

        Returns:
            None
        """
        if ticker not in self.dataframe.columns:
            raise ValueError(f"Ticker '{ticker}' not found in DataFrame columns.")

        returns = self.dataframe[ticker].pct_change().dropna()

        plt.figure(figsize=(12, 6))
        sns.histplot(returns, bins=bins, kde=True)
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.title(f'Return Distribution of {ticker}')
        plt.show()

    def rolling_statistics(self, window=20):
        """
        Compute rolling window statistics (mean, std, variance) for each stock.

        Args:
            window (int): The window size for rolling calculations.

        Returns:
            dict: A dictionary containing DataFrames of rolling statistics.
        """
        rolling_stats = {
            'Rolling_Mean': self.dataframe.rolling(window=window, min_periods=1).mean(),
            'Rolling_Std': self.dataframe.rolling(window=window, min_periods=1).std(),
            'Rolling_Var': self.dataframe.rolling(window=window, min_periods=1).var()
        }
        return rolling_stats

    def plot_rolling_statistics(self, ticker, window=20):
        """
        Plot rolling statistics for a specific ticker.

        Args:
            ticker (str): The ticker symbol of the stock to plot.
            window (int): The window size for rolling calculations.

        Returns:
            None
        """
        if ticker not in self.dataframe.columns:
            raise ValueError(f"Ticker '{ticker}' not found in DataFrame columns.")

        rolling_mean = self.dataframe[ticker].rolling(window=window, min_periods=1).mean()
        rolling_std = self.dataframe[ticker].rolling(window=window, min_periods=1).std()

        plt.figure(figsize=(12, 6))
        plt.plot(self.dataframe.index, self.dataframe[ticker], label='Original')
        plt.plot(self.dataframe.index, rolling_mean, label=f'Rolling Mean ({window})')
        plt.fill_between(self.dataframe.index, rolling_mean - rolling_std, rolling_mean + rolling_std, color='gray', alpha=0.2, label='Rolling Std Dev')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Rolling Statistics of {ticker}')
        plt.legend()
        plt.show()

    def save_statistics(self, stats_df, filename):
        """
        Save computed statistics to a pickle file.

        Args:
            stats_df (pandas.DataFrame): DataFrame containing statistical measures.
            filename (str): The filename to save the statistics.

        Returns:
            None
        """
        PickleHelper(stats_df).pickle_dump(filename)

    def save_outliers(self, outliers_df, filename):
        """
        Save detected outliers to a pickle file.

        Args:
            outliers_df (pandas.DataFrame): DataFrame indicating detected outliers.
            filename (str): The filename to save the outliers.

        Returns:
            None
        """
        PickleHelper(outliers_df).pickle_dump(filename)
