from __future__ import annotations

import time
import warnings

from dataclasses import dataclass
from typing import Dict, Any, Tuple

from IPython.display import display
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

#Random State
RANDOM_STATE = 42

#tf-idf config
TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95
}

#gridsearchCV config
GRID_SEARCH_CONFIG = {
    "cv": 5,
    "n_jobs": -1,
    "verbose": 0,
    "scoring": "f1_weighted"
}

#store hasil eval untuk satu SVM kernel
@dataclass
class SVMResult:

    kernel: str
    model: Any
    best_params: Dict
    accuracy: float
    precision: float
    recall: float
    f1: float
    training_time: float
    confusion_matrix: np.ndarray
    classification_report: Dict

#tfidf builder
def build_vectorizer() -> TfidfVectorizer:

    return TfidfVectorizer(
        max_features=TFIDF_CONFIG["max_features"],
        ngram_range=TFIDF_CONFIG["ngram_range"],
        min_df=TFIDF_CONFIG["min_df"],
        max_df=TFIDF_CONFIG["max_df"]
    )

#Evaluate svm model
def evaluate_model(
    model,
    X_test,
    y_test,
    kernel_name: str,
    training_time: float
) -> SVMResult:
    predictions = model.predict(X_test)

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

    return SVMResult(
        kernel=kernel_name,
        model=model,
        best_params=model.best_params_,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        training_time=training_time,
        confusion_matrix=cm,
        classification_report=report
    )

#grid search for SVM model
def perform_grid_search(
    pipeline: Pipeline,
    parameters: Dict,
    X_train,
    y_train
):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=GRID_SEARCH_CONFIG["cv"],
        scoring=GRID_SEARCH_CONFIG["scoring"],
        n_jobs=GRID_SEARCH_CONFIG["n_jobs"],
        verbose=GRID_SEARCH_CONFIG["verbose"]

    )

    start = time.time()

    grid.fit(
        X_train,
        y_train
    )

    end = time.time()

    training_time = end - start

    return grid, training_time

#Convert kernel to compare table
def create_comparison_table(
    results: Dict[str, SVMResult]
) -> pd.DataFrame:
    
    rows = []

    for kernel, result in results.items():

        rows.append({

            "Kernel": kernel,

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

#Hyperparameter config
LINEAR_PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100]
}


RBF_PARAM_GRID = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", "auto", 0.1, 0.01]
}


POLY_PARAM_GRID = {
    "svm__C": [0.1, 1, 10],
    "svm__degree": [2, 3, 4],
    "svm__gamma": ["scale", "auto"]
}


SIGMOID_PARAM_GRID = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", "auto", 0.1, 0.01]
}

#pipeline builder
def build_pipeline(kernel: str) -> Pipeline:
    pipeline = Pipeline([
        (
            "tfidf",
            build_vectorizer()
        ),
        (
            "svm",
            SVC(
                kernel=kernel,
                random_state=RANDOM_STATE
            )
        )
    ])

    return pipeline

#training linear kernel
def train_linear_kernel(
    X_train,
    X_test,
    y_train,
    y_test
) -> SVMResult:
    print("=" * 60)
    print("Training Linear Kernel")
    print("=" * 60)

    pipeline = build_pipeline("linear")

    grid, training_time = perform_grid_search(
        pipeline,
        LINEAR_PARAM_GRID,
        X_train,
        y_train
    )

    result = evaluate_model(
        grid,
        X_test,
        y_test,
        "Linear",
        training_time
    )

    return result

#RBF kernel training
def train_rbf_kernel(
    X_train,
    X_test,
    y_train,
    y_test
) -> SVMResult:
    print("=" * 60)
    print("Training RBF Kernel")
    print("=" * 60)

    pipeline = build_pipeline("rbf")

    grid, training_time = perform_grid_search(
        pipeline,
        RBF_PARAM_GRID,
        X_train,
        y_train
    )

    result = evaluate_model(
        grid,
        X_test,
        y_test,
        "RBF",
        training_time
    )

    return result

#Polynomial kernel training
def train_polynomial_kernel(
    X_train,
    X_test,
    y_train,
    y_test
) -> SVMResult:
    
    print("=" * 60)
    print("Training Polynomial Kernel")
    print("=" * 60)

    pipeline = build_pipeline("poly")

    grid, training_time = perform_grid_search(
        pipeline,
        POLY_PARAM_GRID,
        X_train,
        y_train
    )

    result = evaluate_model(
        grid,
        X_test,
        y_test,
        "Polynomial",
        training_time
    )

    return result

#Sigmoid kernel training
def train_sigmoid_kernel(
    X_train,
    X_test,
    y_train,
    y_test
) -> SVMResult:
    print("=" * 60)
    print("Training Sigmoid Kernel")
    print("=" * 60)

    pipeline = build_pipeline("sigmoid")

    grid, training_time = perform_grid_search(
        pipeline,
        SIGMOID_PARAM_GRID,
        X_train,
        y_train
    )

    result = evaluate_model(
        grid,
        X_test,
        y_test,
        "Sigmoid",
        training_time
    )

    return result

#Best kernel selection
def get_best_kernel(
    results: Dict[str, SVMResult]
) -> Tuple[str, SVMResult]:
    
    best_kernel = max(
        results,
        key=lambda x: results[x].f1
    )

    return best_kernel, results[best_kernel]

def run_svm(
    X_train,
    X_test,
    y_train,
    y_test
) -> Dict[str, Any]:
    print("=" * 70)
    print("SUPPORT VECTOR MACHINE PIPELINE")
    print("=" * 70)

    results = {}

    #Linar Kernel
    linear_result = train_linear_kernel(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results["Linear"] = linear_result

    #RBF Kernel
    rbf_result = train_rbf_kernel(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results["RBF"] = rbf_result

    #Polynomial Kernel
    poly_result = train_polynomial_kernel(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results["Polynomial"] = poly_result

    #Sigmoid Kernel
    sigmoid_result = train_sigmoid_kernel(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results["Sigmoid"] = sigmoid_result

    #COMPARE
    comparison_table = create_comparison_table(
        results
    )

    best_kernel, best_result = get_best_kernel(
        results
    )

    print("\n")
    print("=" * 70)
    print("KERNEL COMPARISON")
    print("=" * 70)

    print(comparison_table)

    print("\nBest Kernel :", best_kernel)
    print(
        f"Best F1-Score : {best_result.f1:.4f}"
    )

    return {

        "algorithm": "Support Vector Machine",
        "comparison_table": comparison_table,
        "kernel_results": results,
        "best_kernel": best_kernel,
        "best_result": {
            "model": best_result.model,
            "accuracy": best_result.accuracy,
            "precision": best_result.precision,
            "recall": best_result.recall,
            "f1": best_result.f1,
            "training_time": best_result.training_time,
            "best_params": best_result.best_params,
            "classification_report": best_result.classification_report,
            "confusion_matrix": best_result.confusion_matrix
        }
    }

#ALL Report Performance kernel
def print_kernel_summary(
    svm_result: Dict[str, Any]
):
    print("=" * 70)
    print("SVM KERNEL SUMMARY")
    print("=" * 70)

    display(
        svm_result["comparison_table"]
    )

    print("\n")

    print(
        "Best Kernel :",
        svm_result["best_kernel"]
    )

    print(
        "Best Parameters :",
        svm_result["best_result"]["best_params"]
    )