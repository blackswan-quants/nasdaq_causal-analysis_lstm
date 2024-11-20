import numpy as np
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def prepare_lstm_data(df, stock_1, stock_2, input_sequence_length=24, batch_size=32, split_fraction=0.8):
    """
    Prepara i dati per il modello LSTM.
    Divide i dati in train e test set utilizzando una funzione di Keras.
    """
    # Estrai i valori per i due titoli selezionati
    data_stock_1 = df[stock_1].values.reshape(-1, 1)
    data_stock_2 = df[stock_2].values.reshape(-1, 1)

    # Combina i dati in un unico array (stock_1 e stock_2 come feature)
    data = np.hstack([data_stock_1, data_stock_2])

    # Dividi il dataset in training e test
    num_samples = len(data)
    train_size = int(split_fraction * num_samples)

    train_dataset = timeseries_dataset_from_array(
        data=data[:train_size],
        targets=data[input_sequence_length:train_size + input_sequence_length, 0],
        sequence_length=input_sequence_length,
        batch_size=batch_size,
    )
    test_dataset = timeseries_dataset_from_array(
        data=data[train_size:],
        targets=data[train_size + input_sequence_length:, 0],
        sequence_length=input_sequence_length,
        batch_size=batch_size,
    )

    return train_dataset, test_dataset

def train_lstm(train_dataset, input_sequence_length, epochs=10):
    """
    Addestra il modello LSTM sui dati di training.
    """
    # Definizione del modello
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(input_sequence_length, 2)),  # 2 feature (stock_1, stock_2)
        Dense(1)  # Un'uscita scalare (valore predetto)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Addestramento del modello
    model.fit(train_dataset, epochs=epochs)
    return model



def evaluate_model(model, test_dataset, stock_name, output_path="plots/"):
    """
    Valuta il modello sui dati di test e plotta i risultati.
    """
    # Ottieni le previsioni dal modello
    predictions = model.predict(test_dataset)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)

    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label=f"Real {stock_name}")
    plt.plot(predictions, label=f"Predicted {stock_name}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.title(f"Prediction Results for {stock_name}")
    plt.tight_layout()

    # Mostra il grafico
    plt.show()