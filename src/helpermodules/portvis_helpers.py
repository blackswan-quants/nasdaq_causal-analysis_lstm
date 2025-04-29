# Libraries used
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from urllib.request import urlopen
from datetime import datetime
import numpy as np
from twelvedata import TDClient
from dotenv import load_dotenv 
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')
td = TDClient(apikey=API_KEY)

def MDD(portfolio_prices):
    '''
        This function, named MDD (Maximum Drawdown), calculates the maximum 
        drawdown of a portfolio based on the portfolio prices provided as input.
        
        Parameters:
            - portfolio_prices: Dictionary
                A dictionary containing the portfolio value with corresponding dates as indices.
        
        Return:
            - float
                Maximum Drawdown (%)
    '''
    value=list(portfolio_prices.values())
    mdd=0
    ma=-1
    for x in value:
        ma=max(ma, x)
        if mdd < ma-x:
            mdd=ma-x
            mdd_ma=ma
    return (mdd/mdd_ma)*100

def createURL(url, name):
    ''' 
    Given a url and name of an index it creates the correspondant url
    Parameters: url, name (Strings)
    Returns: url (String)
    '''

    for word in name.split():
        if '®' in word:
            word=word[0:-1]

        # & letter cannot be part of a link, %26 to substitute
        if word == "S&P":
            word = "S%26P"

        url += word + "%20"
    return url[:-3] + ".csv"

def get_index_prices(name, ticker):
    '''
    Given an index name and the ticker of an ETF that tracks it, the function
    looks for the index data and returns it in a Dataframe format
    Parameters:
    - name: String
    - ticker: String
    Returns:
    - return_data: pandas Dataframe
    '''

    url_list = ["countries/", "curvo/", "countries_small_cap/", "indexes_gross/", "regions_small_cap/"]
    url_base = "https://raw.githubusercontent.com/NandayDev/MSCI-Historical-Data/main/"

    # trying different paths to the find index data
    response = None
    for url_end in url_list:
        url = createURL(url_base + url_end, name)
        try:
            response = urlopen(url)
        except:
            continue
        break

    # if no index found return None
    if response == None:
        return None

    # converting the response data to a pandas Dataframe
    return_data = pd.read_csv(response, sep=",", names=["Date", ticker], skiprows=1)

    # yahoo finance date format is "2024-04-01", whereas the index data we have has a "2024-04" format
    return_data["Date"] += "-01"

    return return_data

def get_first_date_year(all_date):
    '''
    This function, named get_first_date_year, takes as input a list of 
    dates called all_date. The function returns a list of strings 
    representing the first dates of each year present in the all_date list.
    
    Parameters:
        - all_date: Time_stamp array
            List of dates
    '''
    current_year = int(str(all_date[0])[:4])
    date=[]
    for i in range(0, len(all_date)):
        if current_year == int(str(all_date[i])[:4]):
            date.append(str(all_date[i])[:10])
            current_year+=1
    return date
                
class Portfolio:
    '''
    The Portfolio class helps managing lists of different assets. 

    To initialize it you have to input the following attributes:
    - assets: list of Asset class
    - weights: list of floats, where the 1st float corresponds to the weight of 1st asset and so on
    
    The following attributes are computed automatically using the assets and weights list
    - tickers: list of tickes, one for each Asset 
    - index_names: list of etfs corresponding underlying index names (String)
    - df: pandas dataframe
    '''

    def __init__(self, assets, weights):
        self.assets = assets
        self.weights = weights
        self.tickers = self.load_tickers()
        self.index_names = self.load_index_names()
        self.df = self.load_df()

    def load_tickers(self):
        tickers = []

        for asset in self.assets:
            tickers.append(asset.ticker)
        
        return tickers
        
    def load_index_names(self):
        index_names = []

        for asset in self.assets:
            index_names.append(asset.index_name)
       
        return index_names
    
    def load_df(self):
        '''
        Given the portfolio_tickers and index_names lists the function gets the correspondant index name of each ETF.
        Then, it joins the older index data to the newer ETF data month by month (in % change), so that we have more historical data
        for ETFs. It returns the portfolio_prices dataframe with the older index data added to it.

        Parameters:
        - portfolio_tickers [List of Strings]
        - index_names [List of Strings]

        Returns:
        - portfolio_prices [Dataframe], dataframe with % change of each portfolio component month by month
        '''
        portfolio_tickers = self.tickers
        index_names = self.index_names

        portfolio_tickers.append("IBM")
        portfolio_prices = yf.download(portfolio_tickers, interval='1mo')['Open']
        portfolio_prices = portfolio_prices.pct_change()
        self.tickers.remove("IBM")

        # if it shows error inside this loop it's probably because an index name could not be found on nanday
        # indexes list, improve search algorithm to find correct index name
        for i in (i for i in range(0,len(index_names)) if index_names[i] != ""): 
            ticker = portfolio_tickers[i]
            return_data = get_index_prices(index_names[i], ticker)
            return_data[ticker] = return_data[ticker].pct_change()

            for i in range(0,len(return_data)):
                return_data.loc[i,"Date"] = datetime.strptime(return_data.loc[i,"Date"], '%Y-%m-%d')

            return_data.set_index("Date", inplace = True)
            portfolio_prices[ticker].fillna(return_data[ticker], inplace = True)
        
        portfolio_prices.drop("IBM", axis=1, inplace = True)
        portfolio_prices.dropna(axis = 0, how = 'all', inplace = True)


        return portfolio_prices
    
    def annual_portfolio_return(self):
        '''
        The function annual_portfolio_return calculates the annual return of 
        a portfolio based on the provided stock symbols and their respective 
        weights. It utilizes historical stock price data downloaded from Yahoo Finance, 
        computes the annual percentage returns for each stock in the portfolio, 
        and then calculates the weighted average portfolio return for each year.
        
        Parameters:
            - portfolio_prices: pandas.DataFrame
                It's a dataframe that contains the opening prices of stocks, 
                with tickers as columns and dates as rows.
            - portfolio_tickers: string array
                Array containing the tickers of all assets in the portfolio.
            - portfolio_weight: float array
                Array containing the weights of the various stocks.
        Returns:
            - pandas.DataFrame
                DataFrame containing the annual returns (%) of the portfolio with 
                the year as the index and the returns as the only column.
        '''
        df = self.df.dropna(how='any')
        first_date_year = get_first_date_year(df.index)
        df = self.portfolio_return_pac(1, 0, 0, False, first_date_year[0], first_date_year[-1])

        # if first year date is not january delete the year 
        if first_date_year[0][-4] != '1':
            first_date_year.pop(0)

        # if last year is current year delete it
        if first_date_year[-1][0:4] == str(datetime.now().year):
            first_date_year.pop(-1)
        
        start = df["key_0"].to_list().index(first_date_year[0][0:7])
        annual_returns = [0]*len(first_date_year)
        capital_df = df["Capital"]

        for year in range(0,len(first_date_year)):
            annual_returns[year] = (capital_df[start + (year+1)*12] - capital_df[start + year*12])/capital_df[start + year*12]

        annual_returns = pd.DataFrame(data=annual_returns, index=first_date_year, columns=["Yield"])
    
        return annual_returns
    
    def monthly_portfolio_return(self):
        '''
        This function outputs a dataframe containing the monthly portfolio return of a list of assets. 
        Parameters:
            - portfolio_prices [Dataframe], containing the monthly(!) value of all assets
            - portfolio_tickers [list of Strings]
            - portfolio_weight [list of floats]
        Returns:
            - month_yield [Dataframe]
        '''
        stocks_yield = self.df.dropna(how='any')
        date=list(stocks_yield.index)
        month_yield = pd.DataFrame(columns=['Yield'])

        # set initial value of each asset equal to the weight in the portfolio (ranging from 0 to 1)
        values = self.weights
        
        for i in range(len(stocks_yield.index)):

            # value of each asset at the beginning of the month
            value = sum(values)

            # for each asset, add the change in percentage of that asset * the asset value
            for j in range(len(self.tickers)):
                values[j] += values[j] * stocks_yield.iloc[i][self.tickers[j]]
            
            # the change will be ginen by: (final value - initial value)/initial value
            change = (sum(values) - value)/value

            # set the portfolio yield for that month equal to change
            month_yield.loc[str(date[i])[:7]] = change

        return month_yield

    def portfolio_return_pac(self, starting_capital, amount, fee, fee_is_in_percentage, startdate, enddate):
        '''
        The portfolio_return_pac function outputs a Dataframe with the monthly value of a portfolio built using a PAC (Piano di Accumulo di Capitale) strategy.
        The user can input a starting_capital (initial amount of money in the portfolio), the amount of money that he/she invests each month and a broker's fee.
        If the fee is a fixed amount for each new contribution the percentage parameter should be set as False. If the fee is based on a percentage of the
        contribution the percentage parameter should be set as True.
        Parameters:
            - self [Portfolio object]
            - starting_capital [int]
            - amount [int]
            - fee [int]
            - fee_is_in_percentage [boolean]
        Returns:
            - capital_df [Dataframe]
        '''
        
        # set variables up
        month_yield = self.monthly_portfolio_return()
        capital = starting_capital
        capital_df = pd.DataFrame(columns=['Capital'])
        month_yield= month_yield.loc[startdate:enddate]
        date=list(month_yield.index)

        for i in range(len(date)):
            # FIXME: if an asset grows more than another one its weight increases --> we need to update weights for each month
            #        on the other hand, the weight of the pac is constant thought time
            # for each month, add the amount variable to the capital and subtract the fee 
            if fee_is_in_percentage:
                capital += amount - amount*fee/100
            else:
                capital += amount - fee

            # update the capital variable according to the portfolio performance that month
            capital += month_yield["Yield"].iloc[i]*capital # /100

            # then, update the capital_df dataframe by filling the corresponding month with the new capital value
            capital_df.loc[str(date[i])[:7]] = capital

        capital_df = capital_df.merge(month_yield, left_on=month_yield.index, right_on=capital_df.index, how="left")

        return capital_df

    def graph_plot(self):
        '''
            This function, named graph_plot, generates a representative 
            graph of the portfolio values based on the dates provided 
            in the portfolio_prices dictionary.
            
            Parameter:
                - portfolio_prices : Dictionary
                    A dictionary containing the portfolio value with corresponding dates as indices.
        '''

        all_date=list(self.df.keys())
        date=get_first_date_year(all_date)
        date.append(str(all_date[len(all_date)-1])[:10])

        plt.style.use("ggplot")
        plt.plot(list(self.df.keys()), list(self.df.values()))
        plt.xticks(date,  rotation=45)
        plt.show()

    def graph_returns_frequency(self):
        # FIXME: inflation and dividends data needed here!, use following format to obtain data with inflation and with inflation + dividends:
        '''
        inflation_df = pd.DataFrame(2, index=np.arange(len(self.df)), columns=["Inflation"]) 
        merged_data = self.df.merge(inflation_df, left_on="Year", right_on="Year", how="left")
        merged_data["Return_Adjusted_to_Inflation"] = self.df[self.ticker] - self.df["Inflation"]
        '''

        # FIXME: using average dividend and inflation as of now
        avg_dividend = 3
        avg_inflation = 2

        
        df = self.annual_portfolio_return()
        df["Yield"] *= 100

        df["Return_Adjusted_to_Inflation"] = df["Yield"] - avg_inflation
        df["Return_Adjusted_to_Both"] = df["Return_Adjusted_to_Inflation"] + avg_dividend

        # Plot return distribution with inflation adjustment
        plt.subplot(1, 2, 1)

        plt.hist(df["Yield"], bins=np.arange(min(df["Yield"]), max(df["Yield"]), 5), color='gray', alpha=0.5, label='No Adjustments', edgecolor='black', linewidth=1.2)
        plt.hist(df["Return_Adjusted_to_Inflation"], bins=np.arange(min(df["Yield"]), max(df["Yield"]), 5), color='darkred', alpha=0.5, label='With Inflation', edgecolor='black', linewidth=1.2)
        plt.title('Return Distribution With Inflation Adjustment')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot return distribution with dividends and inflation adjusted
        plt.subplot(1, 2, 2)
        plt.hist(df["Yield"], bins=np.arange(min(df["Yield"]), max(df["Yield"]), 5), color='gray', alpha=0.5, label='No Adjustments', edgecolor='black', linewidth=1.2)
        plt.hist(df["Return_Adjusted_to_Both"], bins=np.arange(min(df["Yield"]), max(df["Yield"]), 5), color='blue', alpha=0.5, label='With Dividends and Inflation', edgecolor='black', linewidth=1.2)
        plt.title('Return Distribution With Dividends and Inflation Adjustment')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()

        # Printing statistics for returns
        stats_pure = df["Yield"].describe()
        stats_both = df["Return_Adjusted_to_Both"].describe()

        print("Statistics for pure Returns:")
        print(round(stats_pure,2))
        print()
        print("Statistics for Returns with Dividends and Inflation Adjustment:") 
        print(round(stats_both,2))

        plt.tight_layout()
        plt.show()


