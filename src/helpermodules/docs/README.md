# Helper Modules
A repository with the modules containing helper modules for specific tasks within our projects. 
As for the required libraries, they're all listed in `requirements.txt`. This repository is made to be used as a submodule, read more on how to use them [here](https://www.git-tower.com/learn/git/ebook/en/command-line/advanced-topics/submodules/).
Here's an overview of what you'll find in each module: 

## Correlation Study Module

The `correlation_study` module is designed for conducting correlation analysis on stock data. It includes a class named `CorrelationAnalysis` that facilitates the following tasks:

### Initialization

```python
CorrelationAnalysis(dataframe, tickers, start_datetime, end_datetime)
```

Initializes the correlation analysis with the provided DataFrame containing stock data, a list of ticker symbols representing the stocks, and start/end date and time for the data.

### Functionality
- get_correlated_stocks(): Calculates correlation coefficients and p-values for the given stocks within the specified time period and stores the results internally.
- plot_corr_matrix(): Plots a heatmap of the correlation matrix using seaborn and matplotlib, displaying correlations between different stocks.
- corr_stocks_pair(): Identifies the pair of stocks with the maximum correlation coefficient, filters based on p-values, and saves the result. Additionally, it plots the stock price data of the identified pair.
### Attributes
- dataframe: Pandas DataFrame containing the stock data.
- tickers: List of ticker symbols representing the stocks.
- start_datetime: Start date and time of the data.
- end_datetime: End date and time of the data.
- corrvalues: Array containing correlation coefficients.
- pvalues: Array containing p-values.
- winner: List containing ticker symbols of the stocks with the highest correlation coefficient.

## Machine Learning Module

The LSTM model training module provides functionality to preprocess stock data, split it into training and testing sets, and build/train an LSTM (Long Short-Term Memory) model for forecasting.

### Functionality

- hashing_and_splitting(adj_close_stocks_dataframe): Splits the given DataFrame of adjusted close prices into training and testing sets based on checksum hashing.
- xtrain_ytrain(adj_close_stocks_dataframe): Splits the DataFrame into training and testing sets, normalizes the data, and prepares it for LSTM model training.
- lstm_model(xtrain, ytrain): Builds and trains an LSTM model using the training data.

## Asset_helpers Module
The asset_helper module provides functionality to represent and manage financial assets such as ETFs (Exchange-Traded Funds), stocks, and bonds. It includes the Asset class, which encapsulates attributes and methods for working with financial data.

### Initialization
Asset(type, ticker, full_name)

Initializes an asset with basic information:

type (str): The type of the asset (e.g., 'ETF', 'Stock', 'Bond').
ticker (str): The ticker symbol or identifier of the asset.
full_name (str): The full name or description of the asset.

### Functionality
apply_ter(ter):
Adjusts the historical data of the asset based on the Total Expense Ratio (TER).

update_from_html(extraction_type):
Updates asset attributes (e.g., ISIN or TER) by extracting information from HTML files.

extraction_type: Specify 'isin' or 'ter' for the type of data to extract.
update_index_name():
Retrieves and updates the index name tracked by the asset based on its ISIN.

load_df():
Loads the historical data of the asset into a Pandas DataFrame using the Twelve Data API.

load():
Performs all necessary operations to fully initialize the asset: updates ISIN, TER, index name, and loads historical data.

info():
Displays key information about the asset, including its type, ticker, TER, index name, ISIN, and loaded data.

### Attributes
type: Type of the asset (e.g., 'ETF', 'Stock', 'Bond').
ticker: Ticker symbol or identifier of the asset.
full_name: Full name or description of the asset.
df: Pandas DataFrame containing historical data of the asset.
index_name: Name of the index the asset tracks (if applicable).
ter: Total Expense Ratio (TER) of the asset.
isin: International Securities Identification Number (ISIN) of the asset.
### Private Methods
_extract_value_from_html(data, start_pattern, end_pattern):
Extracts a value from HTML data between specified start and end patterns.

## Df_dataretrieval Module

The `df_dataretrieval` module provides tools for retrieving, processing, and managing historical stock price data. The module is designed to interface with data sources like Yahoo Finance and the Twelve Data API, allowing for flexible and robust data acquisition.

### Classes

#### 1. `IndexData_Retrieval`

A class for downloading and processing historical stock price data using Yahoo Finance (`yfinance`) or the Twelve Data API.

##### Initialization
IndexData_Retrieval(filename, link, frequency, years=None, months=None, use_yfinance=False)

- filename (str): Name of the pickle file for saving/loading data.
- link (str): URL to a Wikipedia page with stock exchange data.
- frequency (str): Data frequency (e.g., '1min', '1day', '1wk').
- years (int, optional): Number of years of historical data to retrieve (default: `None`).
- months (int, optional): Number of months of historical data to retrieve (default: `None`).
- use_yfinance (bool, optional): Whether to use Yahoo Finance for data retrieval (default: `False`).

##### Methods

- getdata():  
  Loads stock price data from a pickle file or fetches new data if the file doesn't exist.  
  Returns a `pandas.DataFrame` with stock price data.

- get_stockex_tickers():
  Extracts ticker symbols from the specified Wikipedia page.  
  Returns a list of ticker symbols.

- fetch_data(start_date, end_date, use_yfinance=False): 
  Retrieves historical stock price data from Yahoo Finance or Twelve Data API for the specified date range.  
  - start_date(datetime): Start date of the data retrieval window.  
  - end_date (datetime): End date of the data retrieval window.  
  - use_yfinance (bool): If `True`, uses Yahoo Finance; otherwise, uses the Twelve Data API.  
  Returns a `pandas.DataFrame` with stock price data.

- loaded_df():  
  Fetches and saves historical stock price data for the given time window.  
  Returns a `pandas.DataFrame` with the downloaded data.

- clean_df(percentage):
  Cleans the data by removing columns (tickers) with a specified percentage of missing values.  
  - percentage (float): Threshold percentage of missing values.

##### Attributes

- filename: File name for pickle storage.
- link: URL to a Wikipedia page with stock information.
- df: Pandas DataFrame containing stock price data.
- frequency: Data retrieval frequency (e.g., 'daily', 'weekly').
- tickers: List of ticker symbols.
- years:Number of years of historical data to retrieve.
- months: Number of months of historical data to retrieve.
- use_yfinance:Boolean flag indicating whether to use Yahoo Finance.

---

#### 2. `Timestamping`

A class to generate timestamps for market trading hours, accommodating trading days and skipping weekends.

##### Initialization

```python
Timestamping(start_date, end_date, frequency_minutes=1)
```

- start_date (datetime): The start date for timestamp generation.  
- end_date (datetime): The end date for timestamp generation.  
- frequency_minutes (int, optional): Frequency of timestamp generation in minutes (default: `1`).

##### Methods

- __iter__():  
  Returns the iterator object itself.

- __next__():
  Generates the next timestamp in the sequence.  
  Returns a `datetime` object.


## DTWClustering Module

The `DTWClustering` module is designed for computing sequence similarities using various distance metrics, including Euclidean distance and Dynamic Time Warping (DTW). It supports both unconstrained DTW and DTW with a customizable window size.

### Initialization

```python
DTWClustering(seq1, seq2)
```

Initializes the class with two time series sequences:

- seq1 (numpy array): The first time series.  
- seq2 (numpy array): The second time series.  

Both sequences must meet the following criteria:
- Be numpy row vectors (use `.iloc` and `.values` if starting with a Pandas DataFrame).
- Contain no missing or infinite values.
- Be of type `float` (convert with `seq.astype(float)` if necessary).

### Functionality

- euclidean_distance(): 
  Computes the Euclidean distance between `seq1` and `seq2`.  
  Returns a `float` value representing the distance.  
  Raises a `ValueError` if the sequences have different lengths.

- dtw_no_window():
  Computes the Dynamic Time Warping (DTW) distance between `seq1` and `seq2` without any window constraint.  
  Returns a `float` value representing the DTW distance.

- dtw_with_window(window=60): 
  Computes the DTW distance with a specified window constraint.  
  - window (int): The window size for the DTW algorithm (default: `60`).  
  Returns a `float` value representing the DTW distance.

### Attributes

- seq1: The first time series sequence.  
- seq2: The second time series sequence.  


## Memory Handling Module

The `PickleHelper` module simplifies the serialization and deserialization of Python objects using the `pickle` library. It provides a convenient interface for saving and loading datasets, machine learning models, or any other serializable Python objects.

### Class

#### `PickleHelper`

A helper class for managing object serialization and deserialization.

##### Initialization

```python
PickleHelper(obj)
```

- obj: The Python object to be serialized (e.g., a dataset, model, or dictionary).

##### Methods

- pickle_dump(filename):
  Serializes the object and saves it to a file.  
  - filename (str): The name of the file to save the object to. If the filename does not end with `.pkl`, it is automatically appended.  
  Saves the file in the `./pickle_files/` directory.

- pickle_load(filename) (static): 
  Loads a serialized object from a file and returns it as a `PickleHelper` instance.  
  - filename (str): The name of the file to load the object from. If the filename does not end with `.pkl`, it is automatically appended.  
  Returns a `PickleHelper` object with the deserialized object accessible via the `.obj` attribute.  
  If the file does not exist, prints an error message and returns `None`.


## Portfolio Helpers Module

The **Portfolio Helpers Module** provides a suite of tools for constructing, analyzing, and managing investment portfolios. It includes functions and classes to calculate returns, evaluate risks (e.g., Maximum Drawdown), and visualize performance data.

### Functions

#### `MDD(portfolio_prices)`

Calculates the Maximum Drawdown (MDD) of a portfolio based on historical price data.

- Parameters:
  - `portfolio_prices` (dict): Dictionary with dates as keys and portfolio values as values.

- Returns: 
  - `float`: The maximum drawdown percentage.

---

#### `createURL(url, name)`

Generates a URL string by appending the index name to a base URL.

- Parameters:
  - `url` (str): Base URL.
  - `name` (str): Name of the index.

- Returns:  
  - `str`: Generated URL string.

---

#### `get_index_prices(name, ticker)`

Fetches index price data for a given index name and ETF ticker.

- Parameters:
  - `name` (str): Index name.
  - `ticker` (str): ETF ticker symbol.

- Returns:  
  - `pandas.DataFrame`: DataFrame containing index data.

---

#### `get_first_date_year(all_date)`

Extracts the first date of each year from a list of dates.

- Parameters:
  - `all_date` (list): List of timestamps.

- Returns:
  - `list`: List of strings representing the first date of each year.

---

### Class: `Portfolio`

The `Portfolio` class facilitates portfolio construction, analysis, and visualization. It supports ETFs and other assets by integrating historical data and calculating returns.

#### Initialization

```python
Portfolio(assets, weights)
```

- Parameters:
  - `assets` (list): List of `Asset` objects.
  - `weights` (list): List of floats representing asset weights in the portfolio.

#### Attributes

- `assets`: List of portfolio assets.
- `weights`: List of asset weights.
- `tickers`: List of asset tickers.
- `index_names`: List of underlying index names for ETFs.
- `df`: Pandas DataFrame with portfolio data.

#### Methods

- `load_tickers()`  
  Loads the tickers for all portfolio assets.  
  Returns:  
  - `list`: List of ticker symbols.

- `load_index_names()`  
  Loads the index names for all ETFs in the portfolio.  
  Returns: 
  - `list`: List of index names.

- `load_df()` 
  Constructs a DataFrame containing portfolio prices, integrating ETF and index data.  
  Returns: 
  - `pandas.DataFrame`: DataFrame with monthly percentage changes for each asset.

- `annual_portfolio_return()`  
  Calculates annual portfolio returns based on historical price data.  
  Returns:  
  - `pandas.DataFrame`: DataFrame with annual returns (%) indexed by year.

- `monthly_portfolio_return()`  
  Calculates monthly portfolio returns.  
  Returns:
  - `pandas.DataFrame`: DataFrame with monthly portfolio returns.

- **`portfolio_return_pac(starting_capital, amount, fee, fee_is_in_percentage, startdate, enddate)`**  
  Calculates the monthly value of a portfolio built using a PAC (Piano di Accumulo di Capitale) strategy.  
  Parameters:
  - `starting_capital` (float): Initial portfolio value.
  - `amount` (float): Monthly contribution.
  - `fee` (float): Fee per contribution.
  - `fee_is_in_percentage` (bool): Whether the fee is a percentage of the contribution.
  - `startdate` (str): Start date in the format `YYYY-MM-DD`.
  - `enddate` (str): End date in the format `YYYY-MM-DD`.

  Returns: 
  - `pandas.DataFrame`: DataFrame with portfolio values and monthly returns.

- `graph_plot()` 
  Plots the portfolio value over time.

- `graph_returns_frequency()` 
  Visualizes the return distribution with adjustments for inflation and dividends.  
  - Includes histograms for raw returns, returns adjusted for inflation, and returns adjusted for both inflation and dividends.


## Risks Module

The **Risks Module** provides tools to evaluate key financial metrics for analyzing investments, including the Compound Annual Growth Rate (CAGR), standard deviation of returns, Sharpe Ratio, and risk-free rates.

### Functions

#### `evaluate_cagr(prices)`

Calculates the Compound Annual Growth Rate (CAGR) for a given list of prices.

- Parameters:
  - `prices` (list of float): A list of prices, with each value representing a monthly price.

- Returns: 
  - `float`: The CAGR as a decimal (e.g., 0.05 for 5%).

- Example:
  ```python
  prices = [100, 110, 121]
  cagr = evaluate_cagr(prices)
  print(f"CAGR: {cagr:.2%}")
  ```

---

#### `std(returns_array)`

Calculates the standard deviation of an array of returns.

- Parameters:
  - `returns_array` (list of float): A list of return values.

- Returns:  
  - `float`: The standard deviation of the returns.

- Example:
  ```python
  returns = [0.01, -0.02, 0.015, -0.005]
  standard_deviation = std(returns)
  print(f"Standard Deviation: {standard_deviation:.4f}")
  ```

---

#### `deannualize(annual_rate, periods=365)`

Converts an annual interest rate to a rate for a specified number of periods.

- Parameters:
  - `annual_rate` (float): The annual interest rate.
  - `periods` (int, optional): The number of periods in a year (default: 365).

- Returns: 
  - `float`: The de-annualized interest rate.

- Example:
  ```python
  daily_rate = deannualize(0.05)  # Convert 5% annual rate to daily
  print(f"Daily Rate: {daily_rate:.6f}")
  ```

---

#### `get_risk_free_rate(date=None)`

Fetches the risk-free rate based on 3-month US Treasury bills, with the option to retrieve the daily rate for a specific date.

- Parameters:
  - `date` (optional): A specific date (in a format compatible with DataFrame indexing).

- Returns:  
  - `pandas.DataFrame`: DataFrame with annualized and daily rates.
  - `float`: Daily risk-free rate for the specified date.

- Example:
  ```python
  # Fetch all rates
  risk_free_rates = get_risk_free_rate()
  print(risk_free_rates.head())

  # Fetch the rate for a specific date
  specific_date_rate = get_risk_free_rate("2024-01-01")
  print(f"Risk-Free Rate on 2024-01-01: {specific_date_rate:.6f}")
  ```

---

#### `sharpe_ratio(returns_array)`

Calculates the Sharpe Ratio for a given array of returns.

- Parameters:
  - `returns_array` (list of float): A list of returns.

- Returns:
  - `float`: The Sharpe Ratio.

- Example
  ```python
  returns = [0.01, -0.02, 0.015, -0.005]
  sharpe = sharpe_ratio(returns)
  print(f"Sharpe Ratio: {sharpe:.2f}")
  ```

---

### Notes

1. Dependencies: 
   The module uses libraries such as `pandas`, `numpy`, `yfinance`, and `matplotlib`. Ensure these libraries are installed in your environment.
   
2. Risk-Free Rate Source: 
   The risk-free rate is fetched from 3-month US Treasury bill data using the `^IRX` ticker from Yahoo Finance.

3. Assumptions: 
   - Prices and returns provided as input must be free of missing or infinite values.
   - The input data for `evaluate_cagr` is assumed to represent monthly values.



## Inflation Module

The **Inflation Module** provides tools to manage and apply inflation data to financial portfolios. It includes functions for downloading and processing Consumer Price Index (CPI) data and for adjusting portfolio values based on inflation.

---

### Functions

#### `download_cpi_data(selected_countries=None)`

Downloads and processes Consumer Price Index (CPI) data from a CSV file. The function allows filtering for specific countries.

- Parameters:
  - `selected_countries` (list, optional): A list of country names. If provided, returns CPI data for the specified countries. If not provided, returns data for all available countries (excluding Russia).

- Returns:  
  - `pandas.DataFrame`: A DataFrame with CPI data, where rows represent years and columns represent countries.

- Notes:
  - The function processes a CSV file named `Consumer_Price_Index_CPI.csv` and replaces commas with periods for numerical consistency.
  - Russian data is excluded from the dataset.

- Example:
  ```python
  from inflation import download_cpi_data

  # Download CPI data for specific countries
  cpi_data = download_cpi_data(["United States", "Germany"])
  print(cpi_data)

  # Download CPI data for all countries
  all_cpi_data = download_cpi_data()
  print(all_cpi_data.head())
  ```

---

#### `apply_inflation_on_portfolio(portfolio_df, selected_country)`

Applies inflation adjustments to a portfolio based on the Consumer Price Index (CPI) of a selected country.

- Parameters:
  - `portfolio_df` (pandas.DataFrame): A DataFrame containing portfolio data. It must include:
    - `Amount`: Portfolio value.
    - `Pct Change`: Monthly percentage change.
  - `selected_country` (str): The name of the country whose CPI data will be used.

- Returns:  
  - `pandas.DataFrame`: A new DataFrame with the adjusted portfolio values (`Amount` and `Pct Change`) after accounting for inflation.

- Notes:
  - The inflation adjustment is calculated using monthly CPI data derived by dividing annual CPI values by 12.
  - Dates in the `portfolio_df` must be in the format `YYYY-MM-DD`.

- Example:
  ```python
  from inflation import apply_inflation_on_portfolio

  # Sample portfolio DataFrame
  portfolio_df = pd.DataFrame({
      "Amount": [1000, 1020, 1045],
      "Pct Change": [0.02, 0.025, 0.03]
  }, index=["2023-01-01", "2023-02-01", "2023-03-01"])

  # Adjust the portfolio for inflation in the United States
  adjusted_portfolio = apply_inflation_on_portfolio(portfolio_df, "United States")
  print(adjusted_portfolio)
  ```

---

### Dataset Requirements

- The module relies on a CSV file named `Consumer_Price_Index_CPI.csv` with the following structure:
  - Rows: Countries (as the first column) followed by annual CPI data.
  - Columns: Years as headers for CPI values.
- Data for the "Russian Federation" is automatically excluded.





