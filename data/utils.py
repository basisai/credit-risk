import pickle
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


@st.cache(allow_output_mutation=True)
def load_model(filename: str) -> Any:
    return pickle.load(open(filename, "rb"))


@st.cache(allow_output_mutation=True)
def load_data(
    filename: str,
    sample_size: int = None,
    random_state: int = 0,
) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    if sample_size is None:
        return df
    return df.sample(sample_size, random_state=random_state)


@st.cache(allow_output_mutation=True)
def predict(clf: Any, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(x)[:, 1]


@st.cache
def analysis_data() -> pd.DataFrame:
    return pd.read_csv("data/preds.csv")


def print_model_perf(y_val: np.ndarray, y_pred: np.ndarray) -> str:
    text = ""
    text += "Model accuracy = {:.4f}\n".format(
        metrics.accuracy_score(y_val, y_pred))
    text += "Weighted Average Precision = {:.4f}\n".format(
        metrics.precision_score(y_val, y_pred, average="weighted"))
    text += "Weighted Average Recall = {:.4f}\n\n".format(
        metrics.recall_score(y_val, y_pred, average="weighted"))
    text += metrics.classification_report(y_val, y_pred, digits=4)
    return text
