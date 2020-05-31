import pickle

import pandas as pd
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model(filename):
    return pickle.load(open(filename, "rb"))


@st.cache
def load_data(filename, num_rows=None):
    df = pd.read_parquet(filename)
    if num_rows is not None:
        return df.iloc[:num_rows]
    return df


@st.cache(allow_output_mutation=True)
def predict(clf, x):
    """
    For classification, predict probabilities.
    For regression, predict scores.
    """
    return clf.predict_proba(x)[:, 1]
