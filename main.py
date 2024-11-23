from helpermodules.df_dataretrieval import IndexData_Retrieval
from helpermodules.correlation_study import CorrelationAnalysis
from models.lstm import prepare_lstm_data, train_lstm, evaluate_model
import pandas as pd
import os

df_nasdaq = IndexData_Retrieval(
    filename='cleaned_nasdaq_dataframe',
    index='NASDAQ 100',
    months=2,
    frequency='1m',
    use_yfinance=True
)

# load data, clean data frame (closing stock prices)
df_nasdaq.getdata()
if not os.path.isfile('data/pickle_files/cleaned_nasdaq_dataframe.pkl'):
    df_nasdaq.clean_df(percentage=20)

corr_study = CorrelationAnalysis(df_nasdaq.df, df_nasdaq.tickers).winner_rollingcorrelation()

# Divido i dati in training e test set
# train_dataset, test_dataset = prepare_lstm_data(
#     df_combined, 'Stock_1', 'Stock_2', input_sequence_length=24
# )

# 6. Addestramento del modello LSTM
# model = train_lstm(train_dataset, input_sequence_length=24, epochs=10)

# 7. Valutazione del modello
# evaluate_model(model, test_dataset, 'Stock_1')
