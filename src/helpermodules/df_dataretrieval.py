import numpy as np
from datetime import timedelta, datetime
import os
import pandas as pd
from twelvedata import TDClient
import yfinance as yf  # added for yfinance
import re
from helpermodules.memory_handling import PickleHelper
from dotenv import load_dotenv
import time
from pytickersymbols import PyTickerSymbols
from dateutil.relativedelta import relativedelta  # To subtract months

class IndexData_Retrieval:
    """
    A class for downloading and processing historical stock price data using either the Twelve Data API or Yahoo Finance.

    Parameters:
        filename (str): Name of the pickle file to save or load df.
        index (str): Name of the stock index (e.g., 'S&P 500').
        interval (str): Time self.frequency of historical data to load (e.g., '1min', '1day', '1W').
        self.frequency (str): self.frequency of data intervals ('daily', 'weekly', 'monthly', etc.).
        years (int, optional): Number of years of historical data to load (default: None).
        months (int, optional): Number of months of historical data to load (default: None).
        use_yfinance (bool, optional): If True, uses yfinance for data retrieval, otherwise uses Twelve Data API.

    Methods:
        getdata():
            Loads a dataframe of stock price data from a pickle file if it exists, otherwise creates a new dataframe.
            Returns:
                pandas.DataFrame or None: DataFrame containing stock price data if loaded successfully, otherwise None.

        get_stockex_tickers():
            Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
            Returns:
                List[str]: List of ticker symbols extracted from the specified Wikipedia page.

        fetch_data(start_date, end_date):
            Download historical stock prices for the specified time window and data source.
            Returns:
                pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.

        loaded_df():
            Downloads historical stock price data for the specified time window up to the current date and tickers using the selected data source.
            Returns:
                pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.

        clean_df(percentage):
            Cleans the dataframe by dropping stocks with NaN values exceeding the given percentage threshold.
            The cleaned dataframe is pickled after the operation.
            Returns:
                None
    """
    def __init__(self, filename, index, frequency, years=None, months=None, use_yfinance=True):
        self.filename = filename
        self.index = index
        self.df = pd.DataFrame()
        self.frequency = frequency
        self.tickers = []  
        self.years = years
        self.months = months
        self.use_yfinance = use_yfinance

    def getdata(self):
        """
        Loads a dataframe of stock price data from a pickle file if it exists, otherwise creates a new dataframe.

        This method checks for the existence of a pickle file corresponding to the specified filename. 
        If the pickle file exists, it loads the data from it. If not, it checks for a CSV file and loads 
        data from there if available. If neither file exists, it retrieves the stock tickers and 
        fetches new data.

        Returns:
            pandas.DataFrame or None: DataFrame containing stock price data if loaded successfully, otherwise None.
        """
        # Append .pkl extension to filename if missing
        if not re.search(r"\.pkl$", self.filename):
            self.filename += ".pkl"
        
        # Construct the file paths
        pkl_file_path = os.path.join("data", "pickle_files", self.filename)

        # Corresponding CSV file path (replace .pkl with .csv)
        csv_file_path = os.path.join("data", "files", re.sub(r"\.pkl$", ".csv", self.filename))

        # Check if the pickle file exists to load previously saved data
        if os.path.isfile(pkl_file_path):
            try:
                # Load data from pickle if it exists and set ticker columns
                self.df = PickleHelper.pickle_load(self.filename).obj
                self.tickers = self.df.columns.tolist()
                return self.df
            except Exception as e:
                print(f"Error loading pickle file {self.filename}: {e}")
                return None

        # Check if the CSV file exists
        elif os.path.isfile(csv_file_path):
            try:
                # Load data from CSV if pickle does not exist but CSV exists
                self.df = pd.read_csv(csv_file_path)
                self.tickers = self.df.columns.tolist()
                return self.df
            except Exception as e:
                print(f"Error loading CSV file {self.filename}: {e}")
                return None
        
        else:
            # Get tickers if no saved data, and load a new DataFrame
            self.tickers = self.get_stockex_tickers()
            self.df = self.loaded_df()
            

    def get_stockex_tickers(self):
        """
        Get list of the indexes' tickers using PyTickerSymbols.

        Returns:
            list: List of ticker symbols
        """
        stock_data = PyTickerSymbols()
        tickers = stock_data.get_stocks_by_index(self.index)

        # Check if tickers is not empty
        if tickers:
            print("Tickers have been successfully downloaded.")
            ticker_symbols = [stock['symbol'] for stock in tickers]
            print(f"List of ticker symbols: {ticker_symbols}")
        else:
            print("No tickers were found. Check the index name or data source.")
            ticker_symbols = []
        # Return the list of ticker symbols
        return ticker_symbols


    
    def fetch_data(self):
        """
            Fetches historical stock data for a list of tickers from either Yahoo Finance or the Twelve Data API 
            and stores the data in a DataFrame. This method uses the `use_yfinance` flag to determine the data 
            source, and if Yahoo Finance is used, it fetches the data within a specified date range. If the Twelve 
            Data API is used, it divides the tickers into smaller batches for API calls and handles rate limits 
            by pausing between requests.

            The data is retrieved with the 'Adj Close' column, and the result is stored in a DataFrame, where 
            each column represents the adjusted close price for a particular ticker.

            Parameters:
            -----------
            start_date : datetime
                The starting date for fetching historical data. If you specify the `months` or `years` parameter, 
                the method will compute the start date based on the current date. This is typically not directly passed 
                by the user but derived from the given `months` or `years` parameter.

            end_date : datetime
                The ending date for fetching historical data. It is typically set to the current date or the date the 
                function is called, especially when `use_yfinance` is True.
            use_yfinance : bool, optional
                If True, the function uses Yahoo Finance (`yfinance`) for data retrieval. Otherwise, it uses the 
                Twelve Data API (default is False).
            
            frequency : str, optional
                The frequency of the data points to be retrieved (e.g., '1m', '1h', '1d'). Valid frequencies depend 
                on the data source.

            months : int, optional
                The number of months of historical data to fetch (used only when `use_yfinance` is True).

            Returns:
            --------
            pd.DataFrame
                A DataFrame containing the historical data for the specified tickers. The DataFrame has tickers as 
                columns and the corresponding adjusted closing prices for the specified date range.

            Notes:
            ------
            - If the `use_yfinance` flag is set to True, the data is fetched from Yahoo Finance. The method calculates
            the start and end dates dynamically based on the current date and the number of months specified.
            - If the `use_yfinance` flag is False, the data is fetched using the Twelve Data API. The tickers are divided 
            into batches to avoid hitting the API rate limits, and multiple API calls are made to fetch data in segments 
            if necessary.
            - If the data is too large to fetch in a single request (i.e., more than 5000 data points), the method divides 
            the requests into smaller time windows to comply with API limits.
            """

        if self.use_yfinance:
            print(f"use_yfinance: {self.use_yfinance}")  # Debugging line
                        
            # Calculate today's date (end date)
            end_date = datetime.today()

            # Calculate the start date using months if provided, otherwise use years
            if self.months is not None:
                start_date = end_date - relativedelta(months=self.months)
            elif hasattr(self, 'years') and self.years is not None:
                start_date = end_date - relativedelta(years=self.years)
            else:
                raise ValueError("Neither 'months' nor 'years' is provided. Please set one of them.")

            # Convert the start and end date to the correct format for Yahoo Finance (YYYY-MM-DD)
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            # Use yfinance to fetch data
            # Valid intervals of frequencies
            valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m',
                               '1h', '1d', '5d', '1wk', '1mo', '3mo']
            
            if self.frequency not in valid_intervals:
                raise ValueError(f"Frequency '{self.frequency}' not valid for yfinance.")
            
                    # Initialize list to store the data frames
            data_frames = []
                    
            # Initialize list to store data frames
            data_frames = []
            
            # Check if the frequency is one of the minute intervals
            if self.frequency in ['1m', '2m', '5m', '15m', '30m']:
                # Start with the provided start date
                current_start = start_date
                
                # Loop to fetch 8-day chunks of data until we reach the end date
                while current_start < end_date:
                    # Calculate the end of the current 8-day window
                    current_end = current_start + timedelta(days=8)
                    
                    # Ensure the current_end does not exceed the overall end date
                    if current_end > end_date:
                        current_end = end_date
                    
                    print(f"Fetching data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")

                    # Download data from Yahoo Finance with selected settings
                    data = yf.download(
                        tickers=self.tickers,
                        start=current_start.strftime('%Y-%m-%d'),
                        end=current_end.strftime('%Y-%m-%d'),
                        interval=self.frequency,
                        group_by='ticker',
                        auto_adjust=False,  # Set to False since we'll manually select 'Adj Close'
                        prepost=False,
                        threads=True,
                        proxy=None
                    )
                    
                    # Filter only the 'Adj Close' columns and ensure data is valid
                    if isinstance(data.columns, pd.MultiIndex):
                        # If multi-index, select 'Adj Close' for each ticker
                        data = data.loc[:, (slice(None), 'Adj Close')]
                        data.columns = data.columns.droplevel(1)  # Drop 'Adj Close' level, keeping the ticker as column name
                    else:
                        # If not multi-index, ensure we only select 'Adj Close'
                        data = data[['Adj Close']]
                    
                    # Append the data frame to the list
                    data_frames.append(data)
                    
                    # Update the start date for the next 8-day period
                    current_start = current_end

                    # To respect Yahoo's rate limit, add a small delay between requests
                    time.sleep(1)
                
                # Concatenate all the dataframes for the different chunks
                self.df = pd.concat(data_frames)

            # Return the resulting data frame
                return self.df

            else:
                # Fetch data for other frequencies (e.g., '1h', '1d', etc.) in a single request
                data = yf.download(
                    tickers=self.tickers,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=self.frequency,
                    group_by='ticker',
                    auto_adjust=False,  # Set to False since we'll manually select 'Adj Close'
                    prepost=False,
                    threads=True,
                    proxy=None
                )

                # Filter only the 'Adj Close' columns and ensure data is valid
                if isinstance(data.columns, pd.MultiIndex):
                    # If multi-index, select 'Adj Close' for each ticker
                    data = data.loc[:, (slice(None), 'Adj Close')]
                    data.columns = data.columns.droplevel(1)  # Drop 'Adj Close' level, keeping the ticker as column name
                else:
                    # If not multi-index, ensure we only select 'Adj Close'
                    data = data[['Adj Close']]

                self.df = data

            # Check if data is empty
            if not self.df.empty:
                print("Successfully downloaded the data using Yahoo Finance.")
            else:
                print("No data was returned.")
            
            return self.df          
        else:
            # Initialize Twelve Data API client with API key from environment
            load_dotenv()
            API_KEY = os.getenv('API_KEY')
            td = TDClient(apikey=API_KEY)
            # Create a DataFrame with all ticker columns, filled initially with NaN
            dataframes = pd.DataFrame(np.nan, columns=self.tickers, index=[d for d in Timestamping(start_date, end_date)])
            
            #divide tickers into batches
            def divide_tickers_inbatches(tickers):
                """
                Divides the tickers list into batches of 55.
                Parameters:
                -----------
                tickers : list
                    The list of ticker symbols to be divided.
                Returns:
                --------
                list
                    A list of ticker batches, each containing up to 55 tickers.
                """
                return [tickers[i:i+55] for i in range(0, len(tickers), 55)]

            ticker_batches = divide_tickers_inbatches(tickers=self.tickers)

            #divide tickers into batches
            def divide_tickers_inbatches(tickers):
                """
                Divides the tickers list into batches of 55.
                Parameters:
                -----------
                tickers : list
                    The list of ticker symbols to be divided.
                Returns:
                --------
                list
                    A list of ticker batches, each containing up to 55 tickers.
                """
                return [tickers[i:i+55] for i in range(0, len(tickers), 55)]

            ticker_batches = divide_tickers_inbatches(tickers=self.tickers)

            # Generate date boundaries for batching if necessary (limit 5000 per batch)
            generator = Timestamping(start_date=start_date, end_date=end_date, frequency_minutes=self.frequency)
            boundaries = []
            timestamps = list(generator)
            if len(timestamps) <= 5000:
                # If data points are <= 5000, no need for batching
                boundaries = [(start_date, end_date)]
            else:
                # Split data into 5000-long boundaries for each API call
                boundary_start = timestamps[0]
                for i in range(0, len(timestamps), 5000):
                    boundary_end = timestamps[min(i + 4999, len(timestamps) - 1)]
                    boundaries.append((boundary_start, boundary_end))
                    boundary_start = timestamps[min(i + 5000, len(timestamps) - 1)]

            # Fetch data for each batch of tickers and boundaries
            for i, ticker_list in enumerate(divide_tickers_inbatches(self.tickers)):
                print(f'Processing batch {i + 1}/{len(ticker_batches)}')
                for ticker in ticker_list:
                    if len(boundaries) == 1:
                        # Single batch if within 5000 limit
                        call_start, call_end = boundaries[0]
                        print(f'Fetching single batch data for {ticker}')
                        try:
                            df = td.time_series(
                                symbol=ticker,
                                interval=f"{self.frequency}m",
                                start_date=call_start,
                                end_date=call_end,
                                outputsize=5000,
                                timezone="America/New_York",
                            ).as_pandas()
                            for index, value in df['close'].items():
                                dataframes.loc[index, ticker] = value
                        except Exception as e:
                            print(f"Error fetching data for {ticker}: {e}")
                    else:
                        # Loop for multi-boundary data retrieval when limit exceeded
                        for j, (call_start, call_end) in enumerate(boundaries):
                            print(f'Fetching data for {ticker} - Call {j + 1}/{len(boundaries)}')
                            try:
                                df = td.time_series(
                                    symbol=ticker,
                                    interval=f"{self.frequency}m",
                                    start_date=call_start,
                                    end_date=call_end,
                                    outputsize=5000,
                                    timezone="America/New_York",
                                ).as_pandas()
                                for index, value in df['close'].items():
                                    dataframes.loc[index, ticker] = value
                            except Exception as e:
                                print(f"Error fetching data for {ticker} - Call {j + 1}/{len(boundaries)}: {e}")
                if len(ticker_batches) == 55:
                    print('Please wait 60 seconds.')
                    time.sleep(60)  # API limit management
                
            return dataframes


    def loaded_df(self):
        # Calculate months of data to retrieve based on either years or months
        if self.years is not None and self.months is None:
            time_window_months = self.years * 12
        elif self.months is not None and self.years is None:
            time_window_months = self.months
        else:
            raise ValueError("Specify either 'years' or 'months', not both.")
        
        # Define the time window from end_date and start_date
        end_date = datetime.now() - timedelta(days=30)
        start_date = end_date - pd.DateOffset(months=time_window_months)
        # Fetch data within the time window
        stocks_df = self.fetch_data()
        if stocks_df is not None:
            PickleHelper(obj=stocks_df).pickle_dump(filename=self.filename)
            print("Successfully loaded the data into a pickle.")
            return stocks_df
        else:
            print("Unable to retrieve data.")
            return None

    def clean_df(self, percentage):
        """
        Cleans the DataFrame by dropping tickers with excessive NaN values and filling remaining NaNs.

        Parameters:
        -----------
        percentage : float
            The threshold percentage of NaN values allowed per ticker. Columns with more than this percentage of NaN will be dropped.

        Returns:
        --------
        None
        """
        
        # Ensure percentage is in decimal form
        if percentage > 1:
            percentage = percentage / 100

        # Drop tickers with NaN above specified threshold
        for ticker in self.tickers:
            count_nan = self.df[ticker].isnull().sum()
            if count_nan > (len(self.df) * percentage):
                self.df.drop(ticker, axis=1, inplace=True)

        # Fill remaining NaNs
        self.df.fillna(method='ffill', inplace=True)  # Forward-fill for existing gaps
        self.df.fillna(method='bfill', inplace=True)  # Backward-fill for leading gaps

        # Handle cases where entire columns may still remain NaN
        self.df.dropna(axis=1, inplace=True, how='all')

        # Save cleaned DataFrame
        PickleHelper(obj=self.df).pickle_dump(filename=f'cleaned_{self.filename}')
        print("Successfully pickled the cleaned df.")


class Timestamping:
    """
    A class that generates timestamps within a specified range, with custom market hours and frequency.

    This class can be used to generate timestamps between a start and end date, considering market hours (9:45 AM to 3:15 PM).
    The frequency of the generated timestamps can be customized in minutes.

    Attributes:
    -----------
    market_open_hour : int
        The hour when the market opens (default is 9).
    market_open_minute : int
        The minute when the market opens (default is 45).
    market_close_hour : int
        The hour when the market closes (default is 15).
    market_close_minute : int
        The minute when the market closes (default is 15).
    current : datetime
        The current timestamp being iterated over, initialized to the start date with market open time.
    end : datetime
        The end date up to which timestamps are generated.
    frequency : int
        The frequency in minutes between each timestamp.

    Methods:
    --------
    __iter__():
        Returns the iterator object itself.
    
    __next__():
        Returns the next timestamp in the sequence, considering market hours and skipping weekends.
        If the current timestamp exceeds the market close time, it advances to the next market day.
        If the current timestamp exceeds the end date, raises StopIteration to terminate the iteration.
    """

    def __init__(self, start_date: datetime, end_date: datetime, frequency_minutes=1):
        """
        Initializes the Timestamping object with the specified start date, end date, and frequency.

        Parameters:
        -----------
        start_date : datetime
            The starting datetime for generating timestamps.
        end_date : datetime
            The ending datetime for generating timestamps.
        frequency_minutes : int, optional
            The frequency of timestamps in minutes (default is 1 minute).
        """
        # Set market open/close times
        self.market_open_hour = 9
        self.market_open_minute = 45
        self.market_close_hour = 15
        self.market_close_minute = 15
        # Initialize starting point and end date for iteration
        self.current = start_date.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
        self.end = end_date
        self.frequency = frequency_minutes

    def __iter__(self):
        """
        Returns the iterator object itself, enabling iteration over timestamps.
        """
        return self

    def __next__(self) -> datetime:
        """
        Returns the next timestamp in the sequence, considering market hours and frequency.

        If the current timestamp exceeds the market close time, the timestamp is advanced to the next market day.
        If the current timestamp exceeds the end date, a StopIteration exception is raised to terminate the iteration.

        Returns:
        --------
        datetime
            The next valid timestamp.

        Raises:
        -------
        StopIteration
            If the current timestamp exceeds the specified end date.
        """
        # Move forward by frequency, check if within market hours
        self.current += timedelta(minutes=self.frequency)
        if self.current.minute > self.market_close_minute and self.current.hour >= self.market_close_hour:
            # Advance to next day at market open if past close
            self.current += timedelta(days=1)
            self.current = self.current.replace(hour=self.market_open_hour, minute=self.market_open_minute)
        if self.current.weekday() == 5:
            self.current += timedelta(days=2)  # Skip Saturday
        if self.current.weekday() == 6:
            self.current += timedelta(days=1)  # Skip Sunday
        if self.current > self.end:
            raise StopIteration
        return self.current
