import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM
import numpy as np
from tensorflow.keras.optimizers import Adam


def initialize_model_cnn():

    model = Sequential()

    model.add(layers.Masking(mask_value=0.0, input_shape=(60, 200)))

    model.add(layers.Conv1D(20, kernel_size=3))

    model.add(layers.Conv1D(20, kernel_size=3))

    model.add(layers.Conv1D(20, kernel_size=3))

    model.add(layers.Conv1D(20, kernel_size=3))

    model.add(layers.Flatten())

    model.add(layers.Dense(20, activation='relu'))

    model.add(layers.Dense(10, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model


def initialize_model_rnn1():

    model = Sequential()

    model.add(layers.Masking(mask_value=0.0, input_shape=(60, 200)))

    model.add(Bidirectional(LSTM(20, return_sequences=True)))

    model.add(Bidirectional(LSTM(20, return_sequences=True)))

    model.add(Bidirectional(LSTM(20, return_sequences=False)))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model


def initialize_model_rnn2():

    model = Sequential()

    model.add(layers.Masking(mask_value=0.0, input_shape=(60, 200)))

    model.add(Bidirectional(LSTM(20, return_sequences=True)))

    model.add(Bidirectional(LSTM(20, return_sequences=True)))

    model.add(Bidirectional(LSTM(20, return_sequences=False)))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model


def initialize_model_rnn2_25():

    model = Sequential()

    model.add(layers.Masking(mask_value=0.0, input_shape=(60, 25)))

    model.add(Bidirectional(LSTM(20, return_sequences=True)))

    model.add(Bidirectional(LSTM(20, return_sequences=True)))

    model.add(Bidirectional(LSTM(20, return_sequences=False)))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model


def initialize_model_rnn_big():

    model = Sequential()

    model.add(layers.Masking(mask_value=0.0, input_shape=(60, 200)))

    model.add(Bidirectional(LSTM(200, return_sequences=True)))

    model.add(Bidirectional(LSTM(200, return_sequences=True)))

    model.add(Bidirectional(LSTM(200, return_sequences=False)))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model
