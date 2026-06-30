from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Dict, Any

#Dataset Overview
def dataset_overview(
    df: pd.DataFrame
) -> Dict[str, Any]:
    overview = {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Data Types": df.dtypes,
        "Head": df.head(),
        "Tail": df.tail()
    }
    return overview

#Missing Value
def missing_value_analysis(
    df: pd.DataFrame
) -> pd.DataFrame:
    result = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage":
            (df.isnull().sum() / len(df)) * 100
    })

    return result

#Duplicate Data
def duplicate_analysis(
    df: pd.DataFrame
) -> Dict[str, int]:
    duplicate = int(
        df.duplicated().sum()
    )

    return {
        "Duplicate": duplicate,
        "Unique": len(df) - duplicate
    }

def data_type_analysis(
    df: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values
    })

def review_length_analysis(
    df: pd.DataFrame
) -> pd.DataFrame:
    review_length = df["text"].astype(str).str.len()

    return review_length.describe().to_frame(
        name="Review Length"
    )

def rating_distribution(
    df: pd.DataFrame
) -> pd.DataFrame:
    rating = (
        df["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating.columns = [
        "Rating",
        "Total"
    ]

    rating["Percentage"] = (
        rating["Total"]
        / rating["Total"].sum()
    ) * 100

    return rating

#Category Distribution
def category_distribution(
    df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    category = (
        df["category"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    category.columns = [
        "Category",
        "Total"
    ]

    return category

def basic_statistics(
    df: pd.DataFrame
) -> pd.DataFrame:
    return df.describe(
        include="all"
    ).transpose()

#create sentiment label
def create_sentiment_label(
    rating: int
) -> str:
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"
    
def apply_sentiment_label(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = (
        df["rating"]
        .apply(create_sentiment_label)
    )
    return df

#Label Encode
LABEL_MAPPING = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

def label_encoding(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    df["label"] = (
        df["sentiment"]
        .map(LABEL_MAPPING)
    )
    return df

#Distribusi Sentimen
def sentiment_distribution(
    df: pd.DataFrame
) -> pd.DataFrame:
    sentiment = (
        df["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment.columns = [
        "Sentiment",
        "Total"
    ]

    sentiment["Percentage"] = (
        sentiment["Total"]
        / sentiment["Total"].sum()
    ) * 100

    return sentiment

#Prepare Target
def prepare_target(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = apply_sentiment_label(df)
    df = label_encoding(df)
    return df

#Run Complete EDA
def run_eda(
    df: pd.DataFrame
) -> Dict[str, Any]:
    overview = dataset_overview(df)
    missing = missing_value_analysis(df)
    duplicate = duplicate_analysis(df)
    datatype = data_type_analysis(df)
    review_length = review_length_analysis(df)
    rating = rating_distribution(df)
    category = category_distribution(df)
    statistics = basic_statistics(df)
    df_target = prepare_target(df)
    sentiment = sentiment_distribution(df_target)

    return {
        "dataset": df,
        "dataset_with_label": df_target,
        "overview": overview,
        "missing": missing,
        "duplicate": duplicate,
        "datatype": datatype,
        "review_length": review_length,
        "rating_distribution": rating,
        "category_distribution": category,
        "basic_statistics": statistics,
        "sentiment_distribution": sentiment
    }

#Ringkasan EDA
def print_eda_summary(
    eda_result: Dict[str, Any]
):
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print("Shape :",
          eda_result["overview"]["Shape"])

    print("\nColumns :")
    print(eda_result["overview"]["Columns"])
    print("\n")

    print("=" * 70)
    print("MISSING VALUE")
    print("=" * 70)

    print(
        eda_result["missing"]
    )

    print("\n")

    print("=" * 70)
    print("DUPLICATE")
    print("=" * 70)

    print(
        eda_result["duplicate"]
    )

    print("\n")

    print("=" * 70)
    print("REVIEW LENGTH")
    print("=" * 70)

    print(
        eda_result["review_length"]
    )

    print("\n")

    print("=" * 70)
    print("RATING DISTRIBUTION")
    print("=" * 70)

    print(
        eda_result["rating_distribution"]
    )

    print("\n")

    print("=" * 70)
    print("CATEGORY DISTRIBUTION")
    print("=" * 70)

    print(
        eda_result["category_distribution"]
    )

    print("\n")

    print("=" * 70)
    print("SENTIMENT DISTRIBUTION")
    print("=" * 70)

    print(
        eda_result["sentiment_distribution"]
    )