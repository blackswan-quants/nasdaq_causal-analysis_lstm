from statsmodels.tsa.stattools import adfuller

def adf_test(series, col_name):
    """
    Esegue il test Augmented Dickey-Fuller per verificare la stazionarietà della serie temporale.

    Args:
        series (pandas.Series): La serie temporale da testare.
        col_name (str): Il nome della colonna per i risultati.
    
    Returns:
        p_value (float): Il valore p del test ADF.
    """
    result = adfuller(series)
    p_value = result[1]
    print(f"Colonna {col_name}: ADF Statistic = {result[0]}, p-value = {p_value}")
    if p_value < 0.05:
        print(f"{col_name} è stazionaria.")
    else:
        print(f"{col_name} non è stazionaria.")
    return p_value
