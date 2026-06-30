from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

from sklearn.metrics import (
    ConfusionMatrixDisplay
)

plt.style.use("default")

sns.set_theme(
    style="whitegrid",
    palette="deep"
)

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10

#Confusion Matrix
def plot_confusion_matrix(
    confusion_matrix,
    labels,
    title: str
):
    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=labels
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.show()

def plot_svm_kernel_comparison(
    comparison_table: pd.DataFrame
):
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12,8)
    )

    axes = axes.flatten()

    for ax, metric in zip(
        axes,
        metrics
    ):
        sns.barplot(
            data=comparison_table,
            x="Kernel",
            y=metric,
            ax=ax
        )
        ax.set_title(metric)
        ax.set_ylim(0,1)

    plt.tight_layout()
    plt.show()

def plot_training_time(
    comparison_table: pd.DataFrame
):
    plt.figure(figsize=(7,5))

    sns.barplot(
        data=comparison_table,
        x="Kernel",
        y="Training Time (s)"
    )

    plt.title(
        "SVM Training Time"
    )

    plt.ylabel(
        "Seconds"
    )

    plt.tight_layout()
    plt.show()

#Top TF-IDF Features
def plot_top_tfidf_features(
    model,
    feature_names,
    top_n: int = 20
):
    svm = model.best_estimator_.named_steps["svm"]

    if not hasattr(svm, "coef_"):

        print("Feature importance only available for Linear SVM.")

        return

    importance = np.mean(
        np.abs(svm.coef_),
        axis=0
    )

    indices = np.argsort(
        importance
    )[-top_n:]

    features = np.array(
        feature_names
    )[indices]

    scores = importance[indices]

    plt.figure(figsize=(10,6))
    plt.barh(
        features,
        scores
    )

    plt.title(
        f"Top {top_n} TF-IDF Features"
    )

    plt.xlabel(
        "Average Weight"
    )

    plt.tight_layout()
    plt.show()

#CNN Learning curve
def plot_cnn_learning_curve(
    history
):
    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["accuracy"],
        label="Training"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation"
    )

    plt.title(
        "CNN Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

#CNN Loss curve
def plot_cnn_loss_curve(
    history
):
    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation"
    )

    plt.title(
        "CNN Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

#LSTM Learning Curve
def plot_lstm_learning_curve(
    history
):
    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["accuracy"],
        label="Training"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation"
    )

    plt.title(
        "LSTM Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

#LSTM Loss Curve
def plot_lstm_loss_curve(
    history
):
    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation"
    )

    plt.title(
        "LSTM Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

#Model Compare
def plot_model_comparison(
    comparison_df: pd.DataFrame
):
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12,8)
    )

    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):

        sns.barplot(
            data=comparison_df,
            x="Model",
            y=metric,
            ax=ax
        )

        ax.set_ylim(0,1)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.show()

#Wordclooud by sentiment
def plot_wordcloud_by_sentiment(
    df: pd.DataFrame
):
    sentiments = [
        "Positive",
        "Neutral",
        "Negative"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18,6)
    )

    for ax, sentiment in zip(
        axes,
        sentiments
    ):

        text = " ".join(

            df[
                df["sentiment"] == sentiment
            ]["clean_text"]
        )

        if len(text.strip()) == 0:
            ax.axis("off")
            ax.set_title(sentiment)
            continue

        cloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        ax.imshow(
            cloud,
            interpolation="bilinear"
        )

        ax.axis("off")
        ax.set_title(sentiment)

    plt.tight_layout()
    plt.show()

#Class ditribution
def plot_class_distribution(
    df: pd.DataFrame
):
    plt.figure(figsize=(7,5))

    sns.countplot(
        data=df,
        x="sentiment",
        order=[
            "Negative",
            "Neutral",
            "Positive"
        ]
    )

    plt.title(
        "Sentiment Distribution"
    )

    plt.xlabel(
        "Sentiment"
    )

    plt.ylabel(
        "Count"
    )

    plt.tight_layout()
    plt.show()

#Feature Importance (Linear SVM)
def plot_linear_svm_feature_importance(
    model,
    feature_names,
    top_n: int = 20
):
    svm = model.best_estimator_.named_steps["svm"]

    if not hasattr(
        svm,
        "coef_"
    ):

        print(
            "Linear SVM only."
        )
        return

    coef = np.mean(
        svm.coef_,
        axis=0
    )

    index = np.argsort(
        np.abs(coef)
    )[-top_n:]

    features = np.array(
        feature_names
    )[index]
    scores = coef[index]

    colors = [
        "green"
        if score > 0
        else "red"
        for score in scores
    ]

    plt.figure(figsize=(10,6))

    plt.barh(
        features,
        scores,
        color=colors
    )

    plt.title(
        "Top Linear SVM Features"
    )

    plt.xlabel(
        "Coefficient"
    )

    plt.tight_layout()
    plt.show()

#Rating Count
def plot_rating_count(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="rating",
        order=sorted(df["rating"].unique())
    )

    plt.title("Rating Count")
    plt.xlabel("Rating")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

#Rating Pie Chart
def plot_rating_pie(df: pd.DataFrame):
    rating_counts = (
        df["rating"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(7, 7))

    plt.pie(
        rating_counts.values,
        labels=[f"Rating {r}" for r in rating_counts.index],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Rating Distribution")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

#Review Length Distribution
def plot_review_length(df: pd.DataFrame):
    review_length = df["text"].astype(str).str.len()

    plt.figure(figsize=(8, 5))

    sns.histplot(
        review_length,
        bins=40,
        kde=True
    )

    plt.title("Review Length Distribution")
    plt.xlabel("Review Length (characters)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

#Top Category
def plot_top_category(df: pd.DataFrame, top_n: int = 10):
    top_category = (
        df["category"]
        .value_counts()
        .head(top_n)
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        x=top_category.values,
        y=top_category.index
    )

    plt.title(f"Top {top_n} Category")
    plt.xlabel("Count")
    plt.ylabel("Category")

    plt.tight_layout()
    plt.show()