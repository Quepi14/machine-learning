from __future__ import annotations

import time
import warnings

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW

warnings.filterwarnings("ignore")

#random state
RANDOM_STATE = 42

tf.random.set_seed(
    RANDOM_STATE
)

LSTM_CONFIG = {
    "max_words": 20000,
    "max_length": 100,
    "embedding_dim": 128,
    "lstm_units": 128,
    "dropout": 0.5,
    "dense_units": 128,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-3
}

#early stopping for epochs
EARLY_STOPPING = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

@dataclass
class LSTMResult:
    model: Any
    history: Any
    accuracy: float
    precision: float
    recall: float
    f1: float
    training_time: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]

#tokenizer build
def build_tokenizer(
    texts
) -> Tokenizer:

    tokenizer = Tokenizer(
        num_words=LSTM_CONFIG["max_words"],
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(
        texts
    )

    return tokenizer

#sequences
def prepare_sequences(
    tokenizer,
    train_text,
    test_text
):

    X_train = tokenizer.texts_to_sequences(
        train_text
    )

    X_test = tokenizer.texts_to_sequences(
        test_text
    )

    X_train = pad_sequences(
        X_train,
        maxlen=LSTM_CONFIG["max_length"],
        padding="post",
        truncating="post"
    )

    X_test = pad_sequences(
        X_test,
        maxlen=LSTM_CONFIG["max_length"],
        padding="post",
        truncating="post"
    )

    return X_train, X_test

#LSTM Model build
def build_lstm_model(
    vocab_size: int,
    num_classes: int
) -> Model:

    inputs = Input(
        shape=(LSTM_CONFIG["max_length"],)
    )

    x = Embedding(
        input_dim=vocab_size,
        output_dim=LSTM_CONFIG["embedding_dim"],
        input_length=LSTM_CONFIG["max_length"]
    )(inputs)

    # Tidak memerlukan reshape.
    # Output Embedding sudah berbentuk:
    # (batch, sequence_length, embedding_dimension)

    x = LSTM(
        units=LSTM_CONFIG["lstm_units"]
    )(x)

    x = Dropout(
        LSTM_CONFIG["dropout"]
    )(x)

    x = Dense(
        LSTM_CONFIG["dense_units"],
        activation="relu"
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="LSTM_Model"
    )

    model.compile(
        optimizer=AdamW(
            learning_rate=LSTM_CONFIG["learning_rate"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

#evaluate model
def evaluate_model(
    model: Model,
    X_test,
    y_test,
    history,
    training_time: float
) -> LSTMResult:
    predictions = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    return LSTMResult(
        model=model,
        history=history,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        training_time=training_time,
        confusion_matrix=cm,
        classification_report=report
    )

#Train model
def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    vocab_size: int,
    num_classes: int
) -> LSTMResult:
    print("=" * 60)
    print("Training LSTM Model")
    print("=" * 60)

    model = build_lstm_model(
        vocab_size,
        num_classes
    )

    start = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=LSTM_CONFIG["epochs"],
        batch_size=LSTM_CONFIG["batch_size"],
        callbacks=[EARLY_STOPPING],
        verbose=1
    )

    training_time = time.time() - start

    result = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        history=history,
        training_time=training_time
    )

    return result

#LSTM Pipeline
def run_lstm(
    train_text,
    test_text,
    y_train,
    y_test
):
    print("=" * 70)
    print("LONG SHORT-TERM MEMORY PIPELINE")
    print("=" * 70)

    tokenizer = build_tokenizer(
        train_text
    )

    X_train, X_test = prepare_sequences(
        tokenizer,
        train_text,
        test_text
    )

    vocab_size = min(
        LSTM_CONFIG["max_words"],
        len(tokenizer.word_index) + 1
    )

    num_classes = len(
        np.unique(y_train)
    )

    result = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        vocab_size=vocab_size,
        num_classes=num_classes
    )

    print("\n")
    print("=" * 70)
    print("LSTM TRAINING FINISHED")
    print("=" * 70)

    print(f"Accuracy : {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall   : {result.recall:.4f}")
    print(f"F1-Score : {result.f1:.4f}")

    return {
        "algorithm": "LSTM",
        "tokenizer": tokenizer,
        "best_result": {
            "model": result.model,
            "history": result.history,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "training_time": result.training_time,
            "classification_report": result.classification_report,
            "confusion_matrix": result.confusion_matrix
        }
    }

def print_lstm_summary(
    lstm_result
):
    result = lstm_result["best_result"]

    print("=" * 70)
    print("LSTM PERFORMANCE SUMMARY")
    print("=" * 70)

    print(f"Accuracy       : {result['accuracy']:.4f}")
    print(f"Precision      : {result['precision']:.4f}")
    print(f"Recall         : {result['recall']:.4f}")
    print(f"F1-Score       : {result['f1']:.4f}")
    print(f"Training Time  : {result['training_time']:.2f} seconds")

    print("\nClassification Report")
    print(result["classification_report"])

    print("\nConfusion Matrix")
    print(result["confusion_matrix"])

    