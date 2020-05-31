import pickle

import pandas as pd
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model(filename):
    return pickle.load(open(filename, "rb"))


@st.cache
def load_data(filename, sample_size=None, random_state=0):
    df = pd.read_csv(filename)
    if sample_size is None:
        return df
    return df.sample(sample_size, random_state=random_state)


@st.cache(allow_output_mutation=True)
def predict(clf, x):
    """
    For classification, predict probabilities.
    For regression, predict scores.
    """
    return clf.predict_proba(x)[:, 1]
