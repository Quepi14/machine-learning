from __future__ import annotations

import re
import string
import unicodedata
import warnings

from typing import Dict, List

import emoji
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

warnings.filterwarnings("ignore")

# DOWNLOAD NLTK RESOURCE

nltk.download("punkt")
nltk.download("stopwords")

# SASTRAWI INITIALIZATION

stemmer = StemmerFactory().create_stemmer()

stopword_factory = StopWordRemoverFactory()

stopword_list = set(stopwords.words("indonesian"))

stopword_list.update(
    stopword_factory.get_stop_words()
)

# PREPROCESSING CONFIGURATION

PREPROCESSING_CONFIG = {
    "lowercase": True,
    "remove_url": True,
    "remove_html": True,
    "remove_email": True,
    "remove_mention": True,
    "remove_hashtag": True,
    "remove_number": True,
    "remove_punctuation": True,
    "remove_symbol": True,
    "remove_emoji": True,
    "remove_mojibake": True,
    "remove_non_printable": True,
    "remove_extra_space": True,
    "normalize_slang": True,
    "reduce_repeated_characters": True,
    "remove_stopword": True,
    "stemming": True
}

# REGEX PATTERN

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+"
)

HTML_PATTERN = re.compile(
    r"<.*?>"
)

EMAIL_PATTERN = re.compile(
    r"\S+@\S+"
)

MENTION_PATTERN = re.compile(
    r"@\w+"
)

HASHTAG_PATTERN = re.compile(
    r"#\w+"
)

NUMBER_PATTERN = re.compile(
    r"\d+"
)

MULTIPLE_SPACE_PATTERN = re.compile(
    r"\s+"
)

REPEATED_CHARACTER_PATTERN = re.compile(
    r"(.)\1{2,}"
)

NON_PRINTABLE_PATTERN = re.compile(
    r"[\x00-\x1f\x7f-\x9f]"
)

# MOJIBAKE CHARACTER LIST
MOJIBAKE_PATTERNS = [
    "ðŸ", "ð", "Ã", "â", "œ", "™", "€", "˜", "™", "¢", "£", "¥", "±", "œ…",
    "â€œ", "â€", "â€", "â€™", "â€“", "â€”", "ï", "¿", "½"
]

# SLANG DICTIONARY
SLANG_DICT: Dict[str, str] = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak", "tak": "tidak", "bgt": "banget",
    "bgt": "banget", "bngt": "banget", "dgn": "dengan", "dr": "dari", "krn": "karena", "udh": "sudah", "sdh": "sudah", "blm": "belum",
    "aja": "saja", "aj": "saja", "yg": "yang", "utk": "untuk", "sm": "sama", "sy": "saya", "gw": "saya", "gua": "saya", "gue": "saya",
    "loe": "kamu", "lu": "kamu", "km": "kamu", "kmu": "kamu", "trs": "terus", "trus": "terus", "tp": "tetapi", "tpi": "tetapi", "jg": "juga",
    "org": "orang", "pd": "pada", "dlm": "dalam", "mantul": "mantap", "makasih": "terima kasih", "mksh": "terima kasih", "thx": "terima kasih", "ok": "baik", "oke": "baik"
}

#lowercase
def case_folding(text: str) -> str:
    if pd.isna(text):
        return ""

    return str(text).lower()

#Remove url
def remove_url(text: str) -> str:
    return URL_PATTERN.sub("", text)

#Remove HTML
def remove_html(text: str) -> str:
    return HTML_PATTERN.sub("", text)

#Remove Email
def remove_email(text: str) -> str:
    return EMAIL_PATTERN.sub("", text)

#Remove Mention
def remove_mention(text: str) -> str:
    return MENTION_PATTERN.sub("", text)

#Remove hashtag
def remove_hashtag(text: str) -> str:
    return HASHTAG_PATTERN.sub("", text)

#Remove Number
def remove_number(text: str) -> str:
    return NUMBER_PATTERN.sub("", text)

#Remove Punctuation
def remove_punctuation(text: str) -> str:
    return text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

#Remove symbols
def remove_symbol(text: str) -> str:
    cleaned = []

    for char in text:
        category = unicodedata.category(char)
        if not category.startswith("S"):
            cleaned.append(char)
    return "".join(cleaned)

#Remove Emoji
def remove_emoji(text: str) -> str:
    return emoji.replace_emoji(
        text,
        replace=""
    )

#Remove MojiBake
def remove_mojibake(text: str) -> str:
    for pattern in MOJIBAKE_PATTERNS:
        text = text.replace(
            pattern,
            ""
        )
    return text

#REMOVE NON PRINTABLE
def remove_non_printable(text: str) -> str:
    return NON_PRINTABLE_PATTERN.sub(
        "",
        text
    )

#REMOVE EXTRA SPACE
def remove_extra_space(text: str) -> str:
    return MULTIPLE_SPACE_PATTERN.sub(
        " ",
        text
    ).strip()

#Remove single character
SINGLE_CHARACTER_PATTERN = re.compile(
    r"\b[a-zA-Z]\b"
)

def remove_single_character(text: str) -> str:
    text = SINGLE_CHARACTER_PATTERN.sub(
        "",
        text
    )
    return remove_extra_space(text)

#Clean Text Pipeline
def clean_text(text: str) -> str:
    text = case_folding(text)
    text = remove_url(text)
    text = remove_html(text)
    text = remove_email(text)
    text = remove_mention(text)
    text = remove_hashtag(text)
    text = remove_number(text)
    text = remove_punctuation(text)
    text = remove_symbol(text)
    text = remove_emoji(text)
    text = remove_mojibake(text)
    text = remove_non_printable(text)
    text = remove_extra_space(text)

    return text

def normalize_slang(text: str) -> str:
    words = text.split()

    normalized = [
        SLANG_DICT[word]
        if word in SLANG_DICT
        else word
        for word in words
    ]

    return " ".join(normalized)

#Reduce Repeated Char
def reduce_repeated_characters(text: str) -> str:
    return REPEATED_CHARACTER_PATTERN.sub(
        r"\1",
        text
    )

#Tokenizer
def tokenize(text: str) -> List[str]:
    return word_tokenize(text)

#Stopword Removal
def remove_stopwords(tokens: List[str]) -> List[str]:

    return [
        token
        for token in tokens
        if token not in stopword_list
    ]

#Stemming
def stemming(tokens: List[str]) -> str:
    sentence = " ".join(tokens)

    return stemmer.stem(sentence)

#normalization pipeline
def normalize_text(text: str) -> str:
    text = normalize_slang(text)
    text = reduce_repeated_characters(text)

    return text

#NLP Preprocessing pipeline
def preprocess_text(text: str) -> str:
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    text = stemming(tokens)

    return text

#full preprocessing
def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = (
        df["text"]
        .astype(str)
        .apply(clean_text)
        .apply(normalize_text)
        .apply(preprocess_text)
    )

    return df

def show_preprocessing_examples(
    df: pd.DataFrame,
    n: int = 5
):
    preview = pd.DataFrame({
        "Original": df["text"].head(n),
        "Preprocessed": df["clean_text"].head(n)
    })

    return preview


    