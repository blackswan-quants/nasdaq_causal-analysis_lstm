# module used in project not in course of making anymore sigh
# Libraries used
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from twelvedata import TDClient 
from dotenv import load_dotenv 
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from df_dataretrieval import Timestamping


#TODO: added docstrings + error handling + fixed some spaghetti code + generalized a function

    

class Asset:
    """
    Represents an asset, such as an ETF (Exchange-Traded Fund), with associated data and functionality.

    Attributes:
    - type (str): The type of the asset (e.g., 'ETF', 'Stock', 'Bond').
    - ticker (str): The ticker symbol or identifier of the asset.
    - full_name (str): The full name or description of the asset.
    - df (pandas.DataFrame): The DataFrame containing historical data of the asset.
    - index_name (str): The name of the index that the asset tracks (if applicable).
    - ter (float): The Total Expense Ratio (TER) of the asset.
    - isin (str): The unique ISIN code of the asset (International Securities Identification Number).

    Methods:
    - apply_ter(ter): Apply a specified TER to adjust the asset's historical data.
    - extract_value_from_html(data, start_pattern, end_pattern): Retrieve the informations contained by 
        the given HTML script and located between start_pattern and end_pattern strings.
    - update_from_html(extraction_type): Update asset attributes (e.g., ISIN, TER) by extracting
        information from HTML data based on the specified extraction type ('isin' or 'ter').
    - update_index_name(): Update the index name of the asset by extracting information from a webpage
        based on its ISIN.
    - load_df_YF(start_date,end_date,period): Update the 'df' attribute by fetching datas via yahoo finance
    - load_df_TD(): Update the 'df' attribute fetching monthly datas via twelvedata API.
    - load_ETF(): If an ETF is called, it updates the following attributes ('isin', 'ter', 'index_name') via implicit
        call of the above functions.
    - info() : Print out the asset's attributes.
    """
    
    def __init__(self, type, ticker, full_name):
        self.type = type
        self.ticker = ticker
        self.df = None
        self.full_name = full_name
        self.index_name = None
        self.ter = None
        self.isin = None

    def _extract_value_from_html(self, data, start_pattern, end_pattern):
        """
        Extracts a value from HTML data based on start and end patterns.

        Parameters:
        - data (str): HTML data to search within.
        - start_pattern (str): Pattern indicating the start of the value.
        - end_pattern (str): Pattern indicating the end of the value.

        Returns:
        - str: Extracted value between start and end patterns, or None if not found.
        """
        with open('./very_long_html.txt', 'r') as file:
            data = file.read()
        start_index = data.upper().find(self.full_name.upper(), 0)
        if start_index == -1:
            return None

        value = ""
        index = start_index + len(start_pattern)
        while index < len(data):
            if data[index] == end_pattern:
                break
            value += data[index]
            index += 1

        return value
    
    def apply_ter(self, ter):
        """
        Applies a specified Total Expense Ratio (TER) adjustment to the historical data of the asset.

        Parameters:
        - ter (float): The Total Expense Ratio (TER) to apply.
        
        """
        if self.ticker not in self.df.columns:
            raise ValueError(f"Ticker '{self.ticker}' not found in DataFrame columns.")

        monthly_ter_pct = (ter / 12) / 100
        columns = self.df[self.ticker]
        new_df = columns.apply(lambda x: x - monthly_ter_pct)
        self.df[self.ticker] = new_df


    def update_from_html(self, extraction_type):
        """
        Updates ETF attributes (e.g., ISIN, TER) by extracting information from HTML data.

        Parameters:
        - extraction_type (str): Type of extraction ('isin' or 'ter').

        Returns:
        - str: Extracted value, or None if extraction is unsuccessful.
        """
        if extraction_type not in ['isin', 'ter']:
            raise ValueError("extraction_type must be either 'isin' or 'ter'.")

        with open('./very_long_html.txt', 'r') as file:
            data = file.read().replace('\n', '')

        if extraction_type == 'isin':
            value = self._extract_value_from_html(data, self.full_name, '"')
            self.isin = value
        elif extraction_type == 'ter':
            if self.isin is None:
                raise ValueError("ISIN is required to extract TER.")
            value = self._extract_value_from_html(data, self.isin, '"%')
            self.ter = value


    def update_index_name(self):
        """
        Update the object's attribute 'index_name' with the index tracked by the given asset (most likely ETFs, Funds)

        """

        if self.isin is None:
            raise ValueError("ISIN is required to update index name.")

        url = f"https://www.justetf.com/it/etf-profile.html?isin={self.isin}"
        options = Options()
        options.add_argument("--headless")
        browser = webdriver.Chrome(options=options)
        browser.get(url)

        html = browser.page_source
        self.index_name = self._extract_value_from_html(html, "replica l'indice", '.')

        browser.quit()


    def load_df_TD(self, start_date, end_date, frequency):
        """
        Update the object's attribute 'df' with the time series of the stock's prices 
        (via twelvedata) turned to a dataframe (frequency set to one month)

        """
        load_dotenv()
        API_KEY = os.getenv('API_KEY')
        td = TDClient(apikey=API_KEY)
        # twelve data tickers don't include the name of the exchange (eg: VUAA.MI would simply be VUAA)
        if "." in self.ticker:
            ticker = self.ticker[0:self.ticker.rfind(".")]
        else:
            ticker = self.ticker

        df = td.time_series(
            symbol=ticker,
            interval=frequency,
            start_date=start_date,
            end_date=end_date,
            timezone="America/New_York"
        )

        # Returns pandas.DataFrame
        self.df = df.as_pandas()

   
    def load_df_YF(self, start_date, end_date=datetime.today().strftime('%Y-%m-%d'), interval='1d'):
        """
        Update the object's attribute 'df' with the time series of the stock's prices 
        (via yahoo finance) turned to a dataframe (interval set to one month)

        Parameters:
        -start_date(str): the date we want to start selecting datas from
        -end_date(str): the date we want to stop selecting datas (by default: today)
        -interval(str): the time interval between the observation (calculated as the mean of 
                    each value in the given period) (by default: 1 day)
                    
        """
        list_intervals = ['1m', '2m', '5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

        if interval not in list_intervals:
            raise ValueError(f"Interval '{interval}' not available.\nAvailable intervals are {list_intervals}.")
        df = yf.download(self.ticker , start_date, end_date, interval) #download the datas 
        df.index = pd.to_datetime(df.index) 
        self.df=df
        if self.df.empty:
            print(f"No data found for ticker {self.ticker} in the specified date range.")
        else:
            print(f"Data loaded successfully for ticker {self.ticker}.")

   
   
    def load_ETF(self):
        '''
        If an ETF is given, it updates the object's attributes ('isin', 'ter', 'index_name', 'df') related to the chosen stock
        Otherwise, it updates just the 'df' attribute
        
        '''
        if self.type == 'ETF':
            self.update_from_html('isin')
            self.update_from_html('ter')
            self.update_index_name()
        else:
            print('The given object is not an ETF.')
    
   
   
    def info(self):
        '''
        It prints out the following informations for the chosen stock: 
        ('Full name', 'Ticker', 'Type', 'Ter', 'Index name', 'ISIN' , 'Dataframe')

        '''
        print("Full name: ", self.full_name)
        print("Ticker: ", self.ticker)
        print("Type: ", self.type)
        print("Ter: ", self.ter, "%")
        print("Index name: ", self.index_name)
        print("Isin: ", self.isin)
        print("Dataframe: \n", self.df)

    # I need df_dataretrieval module 


        
            
        