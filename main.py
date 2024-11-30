from helpermodules.df_dataretrieval import IndexData_Retrieval
from helpermodules.correlation_study import CorrelationAnalysis
from helpermodules.nonlin_granger_casuality import NonlinearNNGrangerCausalityAnalysis
from models.lstm import prepare_lstm_data, train_lstm, evaluate_model
import pandas as pd


from helpermodules.granger_casuality import GrangerCausalityAnalysis
from adf import adf_test

# Carica il dataframe dal file pickle
file_path = 'helpermodules/final_dataframe.pkl'
data = pd.read_pickle(file_path)





'''
df_nasdaq = IndexData_Retrieval(
    filename='nasdaq_dataframe',
    index='NASDAQ-100',
    months=2,
    frequency='1min',
    use_yfinance=True
)


# load data, clean data frame (closing stock prices)
data.getdata()

# FIXME: if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
data.clean_df(5)

# TODO: readapt correlation analysis to new format df

corr_study = CorrelationAnalysis(df_nasdaq.df, df_nasdaq.tickers)
corr_study.get_correlated_stocks()
corr_study.corr_stocks_pair()
corr_study.plot_corr_matrix()
'''

# 3. Caricamento dei dataset 
#df_stock_1 = pd.read_pickle("pickle_files/dataframe_stock_1.pkl")  # Primo titolo 
#df_stock_2 = pd.read_pickle("pickle_files/dataframe_stock_2.pkl")  # Secondo titolo 


# Estrai i dati 
df_stock_1 = data[['AdjClose_Stock_1']].copy()
df_stock_2 = data[['AdjClose_Stock_2']].copy()
'''
# 4. Unione dei due DataFrame (basata sull'indice temporale)
df_combined = pd.concat(
    [df_stock_1[['GOOGL']], df_stock_2[['GOOG']]], axis=1
)
df_combined.fillna(method='ffill', inplace=True)  # Riempie i NaN con il valore precedente
df_combined.dropna(inplace=True)  # Rimuove righe che contengono NaN
'''
# 4. Unione dei due DataFrame (basata sull'indice temporale)
df_combined = pd.concat([df_stock_1, df_stock_2], axis=1)

# Rinomina le colonne per maggiore chiarezza
df_combined.columns = ['Stock_1', 'Stock_2']

# 5. Preparazione dei dati per l'LSTM
# Divido i dati in training e test set
train_dataset, test_dataset = prepare_lstm_data(
    df_combined, 'Stock_1', 'Stock_2', input_sequence_length=24
)

# 6. Addestramento del modello LSTM
#model = train_lstm(train_dataset, input_sequence_length=24, epochs=10)

# 7. Valutazione del modello
#evaluate_model(model, test_dataset, 'Stock_1')

# Esegui il test ADF per verificare la stazionarietà delle serie temporali
print("\n--- Test ADF per la stazionarietà ---")
adf_test(df_stock_1['AdjClose_Stock_1'], 'Stock_1')
adf_test(df_stock_2['AdjClose_Stock_2'], 'Stock_2')
# Differenziazione per rendere le serie stazionarie
df_combined['Stock_1_diff'] = df_combined['Stock_1'].diff().dropna()
df_combined['Stock_2_diff'] = df_combined['Stock_2'].diff().dropna()

# Esegui di nuovo il test ADF sulle serie differenziate
print("\n--- Test ADF per la stazionarietà dopo la differenziazione ---")
adf_test(df_combined['Stock_1_diff'].dropna(), 'Stock_1_diff')
adf_test(df_combined['Stock_2_diff'].dropna(), 'Stock_2_diff')
try:
    granger_analysis = GrangerCausalityAnalysis(df_combined, max_lag=5)

    # Calcola la causalità di Granger per tutte le coppie di titoli
    granger_results = granger_analysis.calculate_granger_causality()

    # Vedi risultati del test di causalità di Granger
    print("\n--- Risultati del Test di Causalità di Granger Lineare ---")
    print(granger_results)

    # Identificare le coppie con causalità significativa
    significant_pairs = granger_analysis.significant_causality_pairs(alpha=0.05)

    # Stampare i risultati
    print("\n--- Pairs with Significant Granger Causality ---")
    print(significant_pairs)

except Exception as e:
    print(f"Error during Granger causality linear test: {e}")

# Gestione del test di causalità di Granger Non-Lineare
try:
    nonlinear_analysis = NonlinearNNGrangerCausalityAnalysis(df_combined, max_lag=5)

    # Calcola la causalità di Granger non-lineare per tutte le coppie di titoli
    nonlinear_results = nonlinear_analysis.calculate_nonlinear_nn_causality(
        epochs=[10, 10], learning_rate=[0.01, 0.01], batch_size=64
    )

    # Comunica i risultati del test di causalità non-lineare
    print("\n--- Risultati del Test di Causalità Non-Lineare ---")
    print(nonlinear_results)

    # Identificare le coppie con causalità significativa non-lineare
    significant_nonlinear_pairs = nonlinear_analysis.significant_causality_pairs(alpha=0.05)

    # Stampare i risultati
    print("\n--- Pairs with Significant Non-Linear Granger Causality ---")
    print(significant_nonlinear_pairs)

except Exception as e:
    print(f"Error during Granger causality nonlinear test: {e}")