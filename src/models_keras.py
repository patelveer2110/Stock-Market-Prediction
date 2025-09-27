# src/models_keras.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

def build_lstm(lookback, n_features, units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, input_shape=(lookback, n_features)))
    model.add(Dropout(dropout))
    model.add(Dense(max(16, units//2), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def build_lstm_cnn(lookback, n_features, filters=64, kernel_size=3, lstm_units=64, dropout=0.2):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(lookback, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=max(8, filters//2), kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def fit_keras_model(model, X_train, y_train, X_val, y_val,
                    save_path=None, epochs=100, batch_size=64, patience=10, verbose=1):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(3, patience//2), min_lr=1e-6)
    ]
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        callbacks.append(ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, save_weights_only=False))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    # if saved, load the best checkpoint automatically when using ModelCheckpoint and restore_best_weights
    return model, history
