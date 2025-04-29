def evaluate_cagr(prices):
  """
    Evaluate the Compound Annual Growth Rate (CAGR) given a list of prices.

    The function calculates the CAGR based on the initial and ending values of the provided price list.
    CAGR is a useful measure of growth over multiple time periods.

    Args:
    prices (list of float): A list of prices, where each price corresponds to a monthly value.

    Returns:
    float: The calculated CAGR as a decimal. For example, a CAGR of 5% will be returned as 0.05.
  """
  months = len(prices)
  years = months / 12

  initial_value = prices[0]
  ending_value = prices[-1]

  cagr = ((ending_value / initial_value) ** (1/years)) - 1

  return cagr


def std(returns_array):
  """
    Calculate the standard deviation of a given array of returns.

    The function computes the standard deviation, which measures the amount of variation or dispersion 
    of a set of values. The standard deviation is calculated as the square root of the variance.

    Args:
    returns_array (list of float): A list of returns.

    Returns:
    float: The calculated standard deviation.
  """
  sum = 0.0
  mean = np.mean(returns_array)
  n = len(returns_array)

  for price in returns_array:
    sum += (price - mean) ** 2

  std = math.sqrt(sum/n)

  return std


# de-annualize yearly interest rates
def deannualize(annual_rate, periods=365):
  """
    De-annualize a yearly interest rate.

    The function converts an annual interest rate to a rate for a specified number of periods (default is daily).

    Args:
    annual_rate (float): The annual interest rate to be de-annualized.
    periods (int, optional): The number of periods in a year. Default is 365.

    Returns:
    float: The de-annualized interest rate for the specified number of periods.
  """
  return (1 + annual_rate) ** (1/periods) - 1

def get_risk_free_rate(date = None):
  """
    Get the risk-free rate from 3-month US Treasury bills.

    The function downloads the adjusted closing prices of 3-month US Treasury bills,
    de-annualizes them to get the daily rates, and creates a DataFrame with both annualized and daily rates.
    If a date is provided, it returns the daily rate for that specific date.

    Args:
    date (optional): The specific date for which the daily rate is required. Should be in a format compatible with DataFrame indexing.

    Returns:
    pandas.DataFrame or float: A DataFrame with annualized and daily rates, or the daily rate for the specified date if provided.
  """
  # download 3-month us treasury bills rates
  annualized = yf.download("^IRX")["Adj Close"]
  # de-annualize
  daily = annualized.apply(deannualize)
  # create dataframe
  df = pd.DataFrame({"annualized": annualized, "daily": daily})

  if date:
    return df.iloc[date, 'daily']

  return df



def sharpe_ratio(returns_array):
  """
    Calculate the Sharpe Ratio for a given array of returns.

    The function computes the Sharpe Ratio, which is a measure of risk-adjusted return. It is calculated 
    as the difference between the CAGR of the returns and the risk-free rate, divided by the standard deviation 
    of the returns.

    Args:
    returns_array (list of float): A list of returns.

    Returns:
    float: The calculated Sharpe Ratio.
  """
  cagr = evaluate_cagr(returns_array)
  # Risk free rate dell'ultimo giorno disponibile, ovvero oggi
  risk_free_rate = get_risk_free_rate().iloc[-1, 1]

  # Standard deviation
  standard_deviation = std(returns_array)


  return (cagr - risk_free_rate)/standard_deviation


