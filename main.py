from helpermodules.df_dataretrieval import IndexData_Retrieval
from helpermodules.correlation_study import CorrelationAnalysis
from models.lstm import prepare_lstm_data, train_lstm, evaluate_model
import pandas as pd

# Caricamento e pulizia dati (già presente nel main)
df_nasdaq = IndexData_Retrieval(
    filename='nasdaq_dataframe',
    index='NASDAQ-100',
    months=2,
    frequency='1min'
)

# load data, clean data frame (closing stock prices)
df_nasdaq.getdata()

# FIXME: if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
df_nasdaq.clean_df(5)

# TODO: readapt correlation analysis to new format df
corr_study = CorrelationAnalysis(df_nasdaq.df, df_nasdaq.tickers)
corr_study.get_correlated_stocks()
corr_study.corr_stocks_pair()
corr_study.plot_corr_matrix()

# LSTM Integration
# 3. Caricamento dei dataset 
df_stock_1 = pd.read_pickle("pickle_files/dataframe_stock_1.pkl")  # Primo titolo 
df_stock_2 = pd.read_pickle("pickle_files/dataframe_stock_2.pkl")  # Secondo titolo 


# 4. Unione dei due DataFrame (basata sull'indice temporale)
df_combined = pd.concat(
    [df_stock_1[['GOOGL']], df_stock_2[['GOOG']]], axis=1
)
df_combined.fillna(method='ffill', inplace=True)  # Riempie i NaN con il valore precedente
df_combined.dropna(inplace=True)  # Rimuove righe che contengono NaN

# Rinomina le colonne per maggiore chiarezza
df_combined.columns = ['Stock_1', 'Stock_2']

# 5. Preparazione dei dati per l'LSTM
# Divido i dati in training e test set
train_dataset, test_dataset = prepare_lstm_data(
    df_combined, 'Stock_1', 'Stock_2', input_sequence_length=24
)

# 6. Addestramento del modello LSTM
model = train_lstm(train_dataset, input_sequence_length=24, epochs=10)

# 7. Valutazione del modello
evaluate_model(model, test_dataset, 'Stock_1')
