import pickle

import pandas as pd
import streamlit as st
from sklearn import metrics

from xai_fairness.toolkit_xai import compute_shap, get_explainer, compute_corrcoef


@st.cache(allow_output_mutation=True)
def load_model(filename):
    return pickle.load(open(filename, "rb"))


@st.cache(allow_output_mutation=True)
def load_data(filename, sample_size=None, random_state=0):
    df = pd.read_parquet(filename)
    if sample_size is None:
        return df
    return df.sample(sample_size, random_state=random_state)


@st.cache(allow_output_mutation=True)
def predict(clf, x):
    return clf.predict_proba(x)[:, 1]


@st.cache(allow_output_mutation=True)
def compute_shap_values(clf, x):
    explainer = get_explainer(model=clf, model_type="tree")
    return compute_shap(explainer, x)


@st.cache
def analysis_data():
    return pd.read_csv("data/preds.csv")


def print_model_perf(y_val, y_pred):
    text = ""
    text += "Model accuracy = {:.4f}\n".format(metrics.accuracy_score(y_val, y_pred))
    text += "Weighted Average Precision = {:.4f}\n".format(
        metrics.precision_score(y_val, y_pred, average="weighted"))
    text += "Weighted Average Recall = {:.4f}\n\n".format(
        metrics.recall_score(y_val, y_pred, average="weighted"))
    text += metrics.classification_report(y_val, y_pred, digits=4)
    return text
