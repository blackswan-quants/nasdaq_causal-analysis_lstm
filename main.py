from helpermodules.df_dataretrieval import IndexData_Retrieval
from helpermodules.correlation_study import CorrelationAnalysis
from models.lstm import prepare_lstm_data, train_lstm, evaluate_model

# Caricamento e pulizia dati (già presente nel main)
df_nasdaq = IndexData_Retrieval(
    filename='nasdaq_dataframe',
    link='https://en.wikipedia.org/wiki/Nasdaq-100',
    months=2,
    frequency='1min'
)

# load data, clean data frame (closing stock prices)
df_nasdaq.getdata()

# FIXME: if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
df_nasdaq.clean_df(5)

# TODO: readapt correlation analysis to new format df
corr_study = CorrelationAnalysis(df_nasdaq)
corr_study.get_correlated_stocks()
stock_1, stock_2 = corr_study.corr_stocks_pair()
corr_study.plot_corr_matrix()

# LSTM Task: prepara i dati, addestra ed esegui valutazione
# (Collega il dataset di Mischa quando pronto)
train_dataset, test_dataset = prepare_lstm_data(
    df_nasdaq.dataframe, stock_1, stock_2, input_sequence_length=24
)
model = train_lstm(train_dataset, input_sequence_length=24, epochs=10)
evaluate_model(model, test_dataset, stock_1)