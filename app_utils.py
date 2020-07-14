import pickle

import pandas as pd
import streamlit as st

from xai_fairness.toolkit import compute_shap_values


@st.cache(allow_output_mutation=True)
def load_model(filename):
    return pickle.load(open(filename, "rb"))


@st.cache(allow_output_mutation=True)
def load_data(filename, sample_size=None, random_state=0):
    df = pd.read_csv(filename)
    if sample_size is None:
        return df
    return df.sample(sample_size, random_state=random_state)


@st.cache(allow_output_mutation=True)
def predict(clf, x):
    return clf.predict_proba(x)[:, 1]


@st.cache(allow_output_mutation=True)
def compute_shap(clf, x):
    return compute_shap_values(x, model=clf, model_type="tree")
