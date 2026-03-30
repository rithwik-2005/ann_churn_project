import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime


def build_model(input_dim):
    model = Sequential([
        Dense(128, activation="relu", input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y-%m-%d---%H-%M-%S")
    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, mode="max")
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        callbacks=callbacks
    )
    return history


def save_model(model, path="artifacts/model"):
    model.save(path)
