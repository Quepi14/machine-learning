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
    Conv1D,
    Conv2D,
    MaxPooling1D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    Reshape
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import AdamW

warnings.filterwarnings("ignore")

# RANDOM STATE

RANDOM_STATE = 42

tf.random.set_seed(RANDOM_STATE)

# CNN CONFIGURATION

CNN_CONFIG = {
    "max_words": 20000,
    "max_length": 100,
    "embedding_dim": 128,
    "filters": 128,
    "kernel_size": 3,
    "pool_size": 2,
    "dense_units": 128,
    "dropout": 0.5,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-3
}

# EARLY STOPPING

EARLY_STOPPING = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# RESULT DATACLASS

@dataclass
class CNNResult:
    architecture: str
    model: Any
    history: Any
    accuracy: float
    precision: float
    recall: float
    f1: float
    training_time: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]

#Build Conv1d 
def build_conv1d_model(
    vocab_size: int,
    num_classes: int
) -> Model:
    inputs = Input(
        shape=(CNN_CONFIG["max_length"],)
    )

    x = Embedding(
        input_dim=vocab_size,
        output_dim=CNN_CONFIG["embedding_dim"],
        input_length=CNN_CONFIG["max_length"]
    )(inputs)

    x = Conv1D(
        filters=CNN_CONFIG["filters"],
        kernel_size=CNN_CONFIG["kernel_size"],
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)
    x = MaxPooling1D(
        pool_size=CNN_CONFIG["pool_size"]
    )(x)

    x = Flatten()(x)

    x = Dense(
        CNN_CONFIG["dense_units"],
        activation="relu"
    )(x)

    x = Dropout(
        CNN_CONFIG["dropout"]
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_Conv1D"
    )

    model.compile(
        optimizer=AdamW(
            learning_rate=CNN_CONFIG["learning_rate"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

#Build Conv2d
def build_conv2d_model(
    vocab_size: int,
    num_classes: int
) -> Model:
    inputs = Input(
        shape=(CNN_CONFIG["max_length"],)
    )

    """
    embedding layer
    output shape: (batch_size, max_length, embedding_dim)
    """
    x = Embedding(
        input_dim=vocab_size,
        output_dim=CNN_CONFIG["embedding_dim"],
        input_length=CNN_CONFIG["max_length"]
    )(inputs)

    """
    reshape layer

    conv2d butuh input (height, width, channels)

    sehingga (batch_size, max_length, embedding_dim) harus diubah menjadi (batch_size, max_length, embedding_dim, 1)
    """

    x = Reshape(
        (
            CNN_CONFIG["max_length"],
            CNN_CONFIG["embedding_dim"],
            1
        )
    )(x)

    x = Conv2D(
        filters=CNN_CONFIG["filters"],
        kernel_size=(3,3),
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)
    x = MaxPooling2D(
        pool_size=(2,2)
    )(x)

    x = Flatten()(x)

    x = Dense(
        CNN_CONFIG["dense_units"],
        activation="relu"
    )(x)

    x = Dropout(
        CNN_CONFIG["dropout"]
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_Conv2D"
    )

    model.compile(
        optimizer=AdamW(
            learning_rate=CNN_CONFIG["learning_rate"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

#TOKENIZER
def build_tokenizer(
    texts
) -> Tokenizer:
    tokenizer = Tokenizer(
        num_words=CNN_CONFIG["max_words"],
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer

#Prepare Sequence
def prepare_sequences(
    tokenizer: Tokenizer,
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
        maxlen=CNN_CONFIG["max_length"],
        padding="post",
        truncating="post"
    )

    X_test = pad_sequences(
        X_test,
        maxlen=CNN_CONFIG["max_length"],
        padding="post",
        truncating="post"
    )

    return X_train, X_test

#EVALUATE CNN MODEL
def evaluate_model(
    model: Model,
    X_test,
    y_test,
    architecture: str,
    history,
    training_time: float
) -> CNNResult:
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

    return CNNResult(
        architecture=architecture,
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
    architecture: str,
    X_train,
    X_test,
    y_train,
    y_test,
    vocab_size: int,
    num_classes: int
) -> CNNResult:
    print("=" * 60)
    print(f"Training {architecture}")
    print("=" * 60)

    if architecture.lower() == "conv1d":
        model = build_conv1d_model(
            vocab_size,
            num_classes
        )
    elif architecture.lower() == "conv2d":
        model = build_conv2d_model(
            vocab_size,
            num_classes
        )
    else:
        raise ValueError(
            "Architecture must be Conv1D or Conv2D."
        )

    start = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=CNN_CONFIG["epochs"],
        batch_size=CNN_CONFIG["batch_size"],
        callbacks=[EARLY_STOPPING],
        verbose=1
    )

    training_time = time.time() - start

    result = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        architecture=architecture,
        history=history,
        training_time=training_time
    )

    return result

#Comparation table
def create_comparison_table(
    results: Dict[str, CNNResult]
) -> pd.DataFrame:
    rows = []

    for architecture, result in results.items():

        rows.append({
            "Architecture": architecture,
            "Accuracy": result.accuracy,
            "Precision": result.precision,
            "Recall": result.recall,
            "F1-Score": result.f1,
            "Training Time (s)": result.training_time
        })

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        by="F1-Score",
        ascending=False
    ).reset_index(drop=True)

    return comparison

#Best architecture
def get_best_architecture(
    results: Dict[str, CNNResult]
):
    best_architecture = max(
        results,
        key=lambda x: results[x].f1
    )

    return best_architecture, results[best_architecture]

#CNN PIPELINE
def run_cnn(
    train_text,
    test_text,
    y_train,
    y_test
):
    print("=" * 70)
    print("CNN PIPELINE")
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
        CNN_CONFIG["max_words"],
        len(tokenizer.word_index) + 1
    )

    num_classes = len(
        np.unique(y_train)
    )

    results = {}

    #Conv1D
    conv1d_result = train_model(

        architecture="Conv1D",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        vocab_size=vocab_size,
        num_classes=num_classes
    )

    results["Conv1D"] = conv1d_result

    #Conv2D 
    conv2d_result = train_model(
        architecture="Conv2D",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        vocab_size=vocab_size,
        num_classes=num_classes
    )

    results["Conv2D"] = conv2d_result

    comparison_table = create_comparison_table(
        results
    )

    best_architecture, best_result = get_best_architecture(
        results
    )

    print("\n")
    print("=" * 70)
    print("CNN ARCHITECTURE COMPARISON")
    print("=" * 70)

    print(comparison_table)

    print(f"\nBest Architecture : {best_architecture}")

    print(
        f"Best F1-Score : {best_result.f1:.4f}"
    )

    return {
        "algorithm": "CNN",
        "comparison_table": comparison_table,
        "architecture_results": results,
        "best_architecture": best_architecture,
        "tokenizer": tokenizer,
        "best_result": {
            "model": best_result.model,
            "history": best_result.history,
            "accuracy": best_result.accuracy,
            "precision": best_result.precision,
            "recall": best_result.recall,
            "f1": best_result.f1,
            "training_time": best_result.training_time,
            "classification_report": best_result.classification_report,
            "confusion_matrix": best_result.confusion_matrix
        }
    }