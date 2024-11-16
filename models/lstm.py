import numpy as np
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def prepare_lstm_data(df, stock_1, stock_2, input_sequence_length=24, batch_size=32, split_fraction=0.8):
    """
    Prepara i dati per il modello LSTM.
    - Combina i dati di stock_1, stock_2 e aggiunge la loro correlazione come feature.
    - Divide i dati in train e test set.
    """
    # Estrai i valori per i due titoli selezionati
    data_stock_1 = df[stock_1].values.reshape(-1, 1)
    data_stock_2 = df[stock_2].values.reshape(-1, 1)

    # Calcola la correlazione tra i due titoli e aggiungila come colonna costante
    correlation_column = np.corrcoef(data_stock_1[:, 0], data_stock_2[:, 0])[0, 1]
    correlation_column = np.full_like(data_stock_1, correlation_column)

    # Combina i dati in un unico array
    data = np.hstack([data_stock_1, data_stock_2, correlation_column])

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
    - Utilizza una sequenza temporale come input e prevede il valore successivo.
    """
    # Definizione del modello
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(input_sequence_length, 3)),
        Dense(1)  # Un'uscita scalare (valore predetto)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Addestramento del modello
    model.fit(train_dataset, epochs=epochs)
    return model

def evaluate_model(model, test_dataset, stock_1):
    """
    Valuta il modello sui dati di test.
    - Plotta i risultati delle previsioni rispetto ai valori reali.
    """
    # Ottieni le previsioni del modello
    predictions = model.predict(test_dataset)

    # Estrai i valori reali dai dati di test
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)

    # Plot dei risultati
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label=f'Real {stock_1}')  # Valori reali
    plt.plot(predictions, label=f'Predicted {stock_1}')  # Previsioni
    plt.title(f'Prediction Results for {stock_1}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()