import pandas as pd
import re

def download_cpi_data(selected_countries=None):
    """
    Download CPI data from a CSV file and process it.

    Parameters:
        selected_country (list): A list of countries for which CPI data is requested. If provided,
            the function will return CPI data only for the selected countries. If not provided,
            data for all countries will be returned.

    Returns:
        pandas.DataFrame: CPI data for the selected countries (if provided) or for all countries.
    """
    cpi_df = pd.read_csv('Consumer_Price_Index_CPI.csv', delimiter=";")
    cpi_df = cpi_df.rename(columns={'Unnamed: 0': 'Country'})

    # Use years as indexes
    cpi_df = cpi_df.set_index('Country').T

    # Remove Russian data
    cpi_df = cpi_df.drop('Russian Federation', axis=1)

    years = list(cpi_df.index.values)
    countries = list(cpi_df.columns.values)

    # Convert "," to "." in the dataframe
    for year in years:
        for country in countries:
            el = str(cpi_df.loc[year].at[country])
            floated_value = float(re.sub(",", ".", el))
            cpi_df = cpi_df.replace(el, floated_value)

    # Return all the dataset or a subset of it
    if selected_countries:
        return cpi_df[selected_countries]
    else:
        return cpi_df

def apply_inflation_on_portfolio(portflio_df, selected_country):
    """
    Apply inflation to a portfolio based on the CPI data of a selected country.

    Parameters:
        portflio_df (pandas.DataFrame): The portfolio data with 'Amount' and 'Pct Change' columns.
        selected_country (list): The country for which CPI data is to be applied.

    Returns:
        pandas.DataFrame: The portfolio data with inflation adjustments applied.
    """
    portfolio_with_inflation = pd.DataFrame()
    cpi_data = download_cpi_data(selected_country)

    dates = list(portflio_df.index)

    # Use [:4] because the format is YYYY-MM-DD
    start_year = int(dates[0][:4])
    end_year = int(dates[-1][:4])

    for year in range(start_year, end_year + 1):
        # Divide annual inflation by 12 to get monthly inflation
        monthly_inflation = cpi_data[year] / 12

        # Iterate over all dates, acting only on the relevant year's dates
        for date in dates:
            if date[:4] == str(year):
                amount = portflio_df.loc[date, 'Amount']
                pct_change = portflio_df.loc[date, 'Pct Change']

                portflio_df.at[date, 'Amount'] = amount - amount * monthly_inflation
                portflio_df.at[date, 'Pct Change'] = pct_change - monthly_inflation

    return portflio_df
