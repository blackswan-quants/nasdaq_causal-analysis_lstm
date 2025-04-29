#NOTE: in order to use this module, you also need to import memory_handling

# Libraries used
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import matplotlib.colors
import scipy.stats as ss
from scipy import signal
from statsmodels.tsa.stattools import coint # import from statsmodels.tsa.vector_ar.vecm if it doesn't work
from datetime import timedelta, datetime
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from helpermodules.memory_handling import PickleHelper

class CorrelationAnalysis:
    """
    A class to perform various correlation analyses on stock data, including:
    1. Pearson correlation analysis to identify correlated stocks.
    2. Identification of the top correlated pairs of stocks, along with correlation visualization.
    3. Rolling correlation for the most correlated stock pair, and generation of a combined DataFrame for further analysis.
    
    Attributes:
        dataframe (pandas.DataFrame): DataFrame containing the stock data.
        tickers (list): List of ticker symbols representing the stocks.
        corrvalues (np.ndarray): Matrix of Pearson correlation coefficients for each stock pair.
        pvalues (np.ndarray): Matrix of p-values associated with the correlation coefficients.
        winner (list): The most correlated stock pair.
    """

    def __init__(self, dataframe, tickers):
        """
        Initialize the CorrelationAnalysis object.
        
        Args:
            dataframe (pandas.DataFrame): The DataFrame containing the stock data.
            tickers (list): List of ticker symbols representing the stocks.
            start_datetime (str): Start date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
            end_datetime (str): End date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
        """
        self.dataframe = dataframe
        self.tickers = tickers 
#        self.start_datetime = start_datetime
 #       self.end_datetime = end_datetime
        self.corrvalues = None
        self.pvalues = None
        self.winner = None
    def get_correlated_stocks(self, use_pct_change=False):
        """
        Calculate Pearson correlation coefficients and p-values for the given stocks, saving them into two separate pickle files: 'correlationvalues_array' and 'pvalues_array'.
        
        Parameters:
            use_pct_change (bool): If True, use percentage change instead of raw values.
        
        Returns:
            None
        """
        num_stocks = len(self.tickers)
        corr_values = np.full((num_stocks, num_stocks), np.nan)  # Initialize with NaNs
        pvalue_array = np.full((num_stocks, num_stocks), np.nan)  # Initialize with NaNs

        for i in range(num_stocks):
            for j in range(num_stocks):
                try:
                    # Prepare data based on use_pct_change flag
                    if use_pct_change:
                        vals_i = self.dataframe[self.tickers[i]].pct_change().dropna().to_numpy()
                        vals_j = self.dataframe[self.tickers[j]].pct_change().dropna().to_numpy()
                    else:
                        vals_i = self.dataframe[self.tickers[i]].to_numpy()
                        vals_j = self.dataframe[self.tickers[j]].to_numpy()
                    
                    # Ensure values are numeric
                    vals_i = pd.to_numeric(vals_i, errors='coerce')
                    vals_j = pd.to_numeric(vals_j, errors='coerce')

                    if np.isnan(vals_i).any() or np.isnan(vals_j).any():
                        raise ValueError("Data contains NaN values after conversion to numeric.")

                    # Calculate Pearson correlation
                    r_ij, p_ij = ss.pearsonr(vals_i, vals_j)
                    corr_values[i, j] = r_ij
                    pvalue_array[i, j] = p_ij

                except ValueError as ve:
                    print(f"ValueError for {self.tickers[i]} and {self.tickers[j]}: {ve}")
                except Exception as e:
                    print(f"Unexpected error for {self.tickers[i]} and {self.tickers[j]}: {e}")

        self.corrvalues = corr_values
        self.pvalues = pvalue_array

        # Persist results using PickleHelper
        try:
            PickleHelper(self.corrvalues).pickle_dump('correlationvalues_array')
            PickleHelper(self.pvalues).pickle_dump('pvalues_array')
        except Exception as e:
            print(f"Error saving correlation or p-value arrays: {e}")
            
        return "Pickle files for correlation values and p-values have been saved."
    
    def top3_corrstocks(self):
        """
        Identifies and processes the most correlated stock pairs and the top three most correlated pairs.
        
        This method performs the following steps:
        1. Filters out correlations with p-values greater than 0.05 and the diagonal of the correlation matrix.
        2. Identifies the top three most correlated pairs and saves each pair's details to pickle files.
        3. Identifies the most correlated stock pair and saves it to `self.winner`.
        4. Plots the correlation matrix heatmap and the price comparison for each of the top three pairs.
        
        Additionally:
        - The details of the top three correlated pairs are saved as pickle files.
        - The most correlated pair is stored in `self.winner`.
        
        Returns:
            list: The most correlated pair of stocks.
        """
        
        self.get_correlated_stocks()
        # Validate correlation and p-value matrices
        self._validate_matrix(self.corrvalues, "correlation coefficients")
        self._validate_matrix(self.pvalues, "p-values")

        # Mask correlations with p-values > 0.05 and the diagonal
        filtered_corr = np.where(self.pvalues > 0.05, np.nan, self.corrvalues)
        np.fill_diagonal(filtered_corr, np.nan)

        # Identify indices of the top three correlations
        top_three_indices = np.argsort(filtered_corr.flatten())[-3:][::-1]  # Flatten, sort, and get top 3 indices
        top_pairs = []

        for idx in top_three_indices:
            i, j = divmod(idx, len(self.tickers))
            if not np.isnan(filtered_corr[i, j]):
                pair = [self.tickers[i], self.tickers[j]]
                top_pairs.append(pair)
        
        # Nested function to plot the correlation heatmap
        def _plot_corr_matrix(corr_matrix):
            """
            Plot the correlation matrix heatmap for the given data.
            
            Parameters:
                corr_matrix (np.ndarray): The correlation matrix to be plotted.
            """
            def _generate_custom_colormap():
                def _generate_custom_colormap():
                    """
                    Generate a custom color map for the heatmap based on correlation values.
                    
                    Returns:
                        matplotlib.colors.LinearSegmentedColormap: Custom color map.
                    """
                norm = matplotlib.colors.Normalize(-1, 1)
                colors = [
                    [norm(-1), "red"],
                    [norm(-0.93), "lightgrey"],
                    [norm(0.93), "lightgrey"],
                    [norm(1), "green"]
                ]
                return matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

            # Use the nested function to get the color map
            cmap = _generate_custom_colormap()

            # Plot the heatmap
            plt.figure(figsize=(40, 20))
            seaborn.heatmap(
                pd.DataFrame(corr_matrix, columns=self.tickers, index=self.tickers),
                annot=True,
                cmap=cmap
            )
            plt.title("Correlation Matrix Heatmap", fontsize=16)
            plt.show()

        # Plot the full correlation matrix for visualization
        _plot_corr_matrix(self.corrvalues)

        # Process and save each pair
        for rank, pair in enumerate(top_pairs, 1):
            print(f"Top {rank} correlated pair: {pair} with correlation {filtered_corr[i, j]:.2f}")

            # Save pair details
            file_name = f"df_maxcorr_pair_{rank}"
            PickleHelper(pair).pickle_dump(file_name)
            print(f"Pickle file saved for pair {pair}: {file_name}.pkl")

            # Plot their price comparison
            self.dataframe[pair].plot(figsize=(12, 6), title=f"Price Comparison: {pair}")
            plt.show()

        # Identify the most correlated pair (the winner)
        tmp_arr = self.corrvalues.copy()
        np.fill_diagonal(tmp_arr, 0)  # Exclude self-correlation
        max_corr = np.nanmax(tmp_arr)
        max_indexes = np.where(self.corrvalues == max_corr)
        self.winner = [self.tickers[max_indexes[0][0]], self.tickers[max_indexes[1][0]]]
        
        print(f"Most correlated pair: {self.winner} with correlation: {max_corr}")
        
        return self.winner


    def winner_rollingcorrelation(self):
        """
        Identify the two most correlated stocks and save their data with rolling correlation as a feature.
        
        This method performs the following steps:
        1. Identifies the most correlated stock pair using `top3_corrstocks`.
        2. Computes the rolling correlation for this pair over a given window (default is '1H').
        3. Creates individual DataFrames for both stocks with the rolling correlation as a feature.
        4. Combines these DataFrames into one and saves the final combined DataFrame using `PickleHelper`.
        
        Nested Methods:
        - `rolling_correlation`: Calculates the rolling correlation between two stock prices over a specified window.
        - `generate_feature_dfs`: Creates individual DataFrames for each stock with the rolling correlation as a feature.
        - `generate_combined_df`: Combines both stock DataFrames along with the rolling correlation into a single DataFrame.
        
        Returns:
            pandas.DataFrame: A combined DataFrame containing the two most correlated stocks and the rolling correlation.
        """
        
        # Step 1: Get the top 3 correlated stocks pickled, get winner
        most_correlated_pair = self.top3_corrstocks()
        
        # Step 2: Compute the rolling correlation between two stock tickers
        # Step 2: Compute the rolling correlation between the two most correlated stocks
        def rolling_correlation(stock1, stock2, window='1H'):
            """
            Compute rolling correlation for two stocks over a specified time window of one hour.
            
            Args:
                stock1 (str): The ticker symbol of the first stock.
                stock2 (str): The ticker symbol of the second stock.
                window (str): The size of the rolling time window (default is 1 hour).
            
            Returns:
                pandas.Series: A time series of rolling correlation values.
            """
            # Ensure the DataFrame index is a datetime index
            if not pd.api.types.is_datetime64_any_dtype(self.dataframe.index):
                raise ValueError("DataFrame index must be a datetime index.")
            
            # Ensure the DataFrame is sorted by the index (DatetimeIndex)
            df = self.dataframe.sort_index()
            
            # Calculate rolling correlation over a time window
            rolling_corr = df[stock1].rolling(window=window).corr(df[stock2])
            return rolling_corr
        
        # Step 3: Create individual DataFrames for both stocks with the rolling correlation as a feature
        def generate_feature_dfs(stock1, stock2, window='1H', fillna_method=None):
            """
            Create individual DataFrames for the given stocks, with correlation as a feature.

            Args:
                stock1 (str): The ticker symbol of the first stock.
                stock2 (str): The ticker symbol of the second stock.
                window (str): The size of the rolling time window (default is 1 hour).
                fillna_method (str, optional): The method to fill NaN values in the correlation column. 
                                            Options: 'ffill', 'bfill', or None (default).

            Returns:
                Two DataFrames, one for each stock.
            """
            # Calculate rolling correlation
            rolling_corr = rolling_correlation(stock1, stock2, window)
            
            # Create DataFrames for each stock
            df_stock1 = self.dataframe[[stock1]].copy()
            df_stock1['correlation'] = rolling_corr
            
            df_stock2 = self.dataframe[[stock2]].copy()
            df_stock2['correlation'] = rolling_corr

            # Optionally fill NaN values in the correlation column if fillna_method is provided
            if fillna_method is not None:
                df_stock1['correlation'] = df_stock1['correlation'].fillna(method=fillna_method)
                df_stock2['correlation'] = df_stock2['correlation'].fillna(method=fillna_method)
                
            return df_stock1, df_stock2

        # Step 4: Combine the DataFrames for both stocks
        def generate_combined_df(df_stock1, df_stock2):
            """
            Create a single DataFrame containing the time series for both stocks
            and the rolling correlation as a shared feature.
            
            Args:
                df_stock1 (pandas.DataFrame): DataFrame containing stock1 data and correlation.
                df_stock2 (pandas.DataFrame): DataFrame containing stock2 data and correlation.
            
            Returns:
                pandas.DataFrame: A single DataFrame containing both stocks' data and the correlation.
            """
            combined_df = pd.concat([df_stock1, df_stock2], axis=1)
            return combined_df

        # Step 5: Execute the process
        df_stock1, df_stock2 = generate_feature_dfs(
            stock1=most_correlated_pair[0], 
            stock2=most_correlated_pair[1], 
            window='60min', 
            fillna_method='ffill'
        )    
        df_final = generate_combined_df(df_stock1, df_stock2)

        # Step 6: Save the DataFrames to pickle files
        PickleHelper(df_final).pickle_dump('final_dataframe')
        print("Pickle file saved for the final dataframe: final_dataframe.pkl")
        
        return df_final


    def get_correlation_lags(self, use_pct_change=False):
        """
        Calculate and store cross-correlation lags as vectors in a 3D array for each stock pair (i, j).
        Store the best lag for each correlation in the best_lag 2D array.
        
        Args:
            use_pct_change (bool): If True, use percentage change instead of raw values.

        Returns:
            None
        """
        corr_lags = np.zeros([len(self.tickers), len(self.tickers), self.dataframe.shape[0]*2-1])
        best_lag = np.zeros([len(self.tickers), len(self.tickers)])
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                if use_pct_change:
                    vals_i = self.dataframe[self.tickers[i]].pct_change().dropna().to_numpy()
                    vals_j = self.dataframe[self.tickers[j]].pct_change().dropna().to_numpy()
                else:
                    vals_i = self.dataframe[self.tickers[i]].to_numpy()
                    vals_j = self.dataframe[self.tickers[j]].to_numpy()
                lags_ij = signal.correlation_lags(len(vals_i), len(vals_j), mode="full")
                corr_lags[i, j] = lags_ij
                correlation = signal.correlate(vals_i, vals_j, mode="full")
                best_lag[i, j] = lags_ij[np.argmax(correlation)]
        self.best_lag = best_lag
        self.corr_lags = corr_lags
        PickleHelper(self.corr_lags).pickle_dump('all_lags_array')
        PickleHelper(self.best_lag).pickle_dump('best_lags_array')

    def get_and_plot_correlation_lags(self, use_pct_change=False):
        """
        Calculate and store cross-correlation lags as vectors in a 3D array for each stock pair (i, j).
        Store the best lag for each correlation in the best_lag 2D array, and plot the cointegration matrix heatmap.
        
        Args:
            use_pct_change (bool): If True, use percentage change instead of raw values.

        Returns:
            None
        """
        # Initialize the arrays for storing correlation lags and best lags
        corr_lags = np.zeros([len(self.tickers), len(self.tickers), self.dataframe.shape[0]*2-1])
        best_lag = np.zeros([len(self.tickers), len(self.tickers)])
        
        # Calculate the correlation lags for each pair of tickers
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                if use_pct_change:
                    vals_i = self.dataframe[self.tickers[i]].pct_change().dropna().to_numpy()
                    vals_j = self.dataframe[self.tickers[j]].pct_change().dropna().to_numpy()
                else:
                    vals_i = self.dataframe[self.tickers[i]].to_numpy()
                    vals_j = self.dataframe[self.tickers[j]].to_numpy()

                # Calculate lags and correlations
                lags_ij = signal.correlation_lags(len(vals_i), len(vals_j), mode="full")
                corr_lags[i, j] = lags_ij
                correlation = signal.correlate(vals_i, vals_j, mode="full")
                best_lag[i, j] = lags_ij[np.argmax(correlation)]
        
        # Store the results
        self.best_lag = best_lag
        self.corr_lags = corr_lags
        PickleHelper(self.corr_lags).pickle_dump('all_lags_array')
        PickleHelper(self.best_lag).pickle_dump('best_lags_array')
        
        # Plot the cointegration matrix heatmap
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [
            [norm(-1), "red"],
            [norm(-0.93), "lightgrey"],
            [norm(0.93), "lightgrey"],
            [norm(1), "green"]
        ]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        
        plt.figure(figsize=(40, 20))
        seaborn.heatmap(pd.DataFrame(self.coint_scores, columns=self.tickers, index=self.tickers), annot=True, cmap=cmap)
        plt.show()

    def print_cross_corr(self, threshold: float, max_lag: int, volumes=None):
        """
        Prints the cross-correlation of stocks with a lag up to max_lag.

        This method iterates through each stock in the dataframe, calculates the cross-correlation with all other stocks, 
        and prints the pairs with a cross-correlation greater than or equal to the specified threshold.

        Args:
            threshold (float): The minimum cross-correlation required for a pair to be considered correlated.
            max_lag (int): The maximum lag to consider for cross-correlation analysis.
            volumes (optional): Not used in this implementation.

        Returns:
            None
        """
        for i in range(len(self.dataframe.columns)):
            for j in range(len(self.dataframe.columns)):
                if i != j:
                    corr_list = signal.correlate(self.dataframe[self.tickers[i]], self.dataframe[self.tickers[j]], mode='full')
                    lags = signal.correlation_lags(len(self.dataframe[self.tickers[i]]), len(self.dataframe[self.tickers[j]]))
                    corr_list = corr_list / (len(self.dataframe[self.tickers[i]]) * self.dataframe[self.tickers[i]].std() * self.dataframe[self.tickers[j]].std())
                    
                    # Normalize correlations to the range [0, 1]
                    sc = MinMaxScaler(feature_range=(0, 1))
                    corr_list_scaled = sc.fit_transform(corr_list.reshape(-1, 1)).flatten()
                    
                    for k, corr in enumerate(corr_list_scaled):
                        if abs(lags[k]) <= max_lag and corr >= threshold:
                            print(f"{self.tickers[i]} and {self.tickers[j]} are correlated ({corr}) with lag = {lags[k]}")


        # Helper Methods
    def _validate_tickers(self):
        """Validates that tickers exist in the DataFrame."""
        if not all(ticker in self.dataframe.columns for ticker in self.tickers):
            raise ValueError("Some tickers are missing in the DataFrame columns.")

    def _validate_matrix(self, matrix, name):
        """Validates that a matrix is not None."""
        if matrix is None:
            raise ValueError(f"{name.capitalize()} matrix has not been calculated yet.")

    def _get_values(self, i, j, use_pct_change):
        """Extracts values for analysis."""
        vals_i = self.dataframe[self.tickers[i]].pct_change().dropna().to_numpy() if use_pct_change else self.dataframe[self.tickers[i]].to_numpy()
        vals_j = self.dataframe[self.tickers[j]].pct_change().dropna().to_numpy() if use_pct_change else self.dataframe[self.tickers[j]].to_numpy()
        return vals_i, vals_j

    def _plot_heatmap(self, data, title, center_color=False):
        """Plots a heatmap for given data."""
        cmap = "coolwarm" if center_color else "viridis"
        plt.figure(figsize=(12, 6))
        sns.heatmap(data, annot=True, xticklabels=self.tickers, yticklabels=self.tickers, cmap=cmap, center=0 if center_color else None)
        plt.title(title)
        plt.show()