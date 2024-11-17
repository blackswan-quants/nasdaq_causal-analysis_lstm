import pickle

# Percorso ai file pickle
file_stock_1 = "pickle_files/dataframe_stock_1.pkl"
file_stock_2 = "pickle_files/dataframe_stock_2.pkl"

# Carica e ispeziona i dati
with open(file_stock_1, "rb") as f:
    df_stock_1 = pickle.load(f)
    print("Contenuto di dataframe_stock_1.pkl:")
    print(df_stock_1.info())
    print(df_stock_1.head())

with open(file_stock_2, "rb") as f:
    df_stock_2 = pickle.load(f)
    print("\nContenuto di dataframe_stock_2.pkl:")
    print(df_stock_2.info())
    print(df_stock_2.head())
print(df_stock_1.iloc[100:110])  # Controlla righe dalla 100 alla 110
print(df_stock_2.iloc[100:110])
