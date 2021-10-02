"""
App for FAI.
"""
import pandas as pd
import streamlit as st

from xai_fairness.toolkit_fai import (
    get_aif_metric,
    compute_fairness_measures,
)
from xai_fairness.static_fai import (
    alg_fai,
    fmeasures_chart,
    fairness_notes,
)

from data.utils import load_model, load_data, predict, print_model_perf
from preprocess.constants import FEATURES, TARGET, PROTECTED_FEATURES

METRICS_TO_USE = [
    "Equal opportunity (equal FNR)",
    "Predictive parity (equal PPV)",
    "Statistical parity",
]


@st.cache
def prepare_pred(x_valid, y_valid, debias=False):
    # Load model
    clf = load_model("data/lgb_model.pkl")

    # Predict on val data
    y_prob = predict(clf, x_valid)

    # st.header("Prediction Distributions")
    cutoff = 0.5
    y_pred = (y_prob > cutoff).astype(int)

    if debias:
        raise NotImplementedError

    # Model performance
    text_model_perf = print_model_perf(y_valid, y_pred)

    return y_pred, text_model_perf


def fai(debias=False):
    protected_attribute = st.selectbox(
        "Select protected column.", list(PROTECTED_FEATURES.keys()))

    # Load data
    valid = load_data("data/test.gz.parquet").fillna(0)
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values
    valid_fai = valid[list(PROTECTED_FEATURES.keys())]

    # Get predictions
    y_pred, text_model_perf = prepare_pred(x_valid, y_valid, debias=debias)

    st.header("Model Performance")
    st.text(text_model_perf)

    st.header("Algorithmic Fairness Metrics")
    threshold = st.slider(
        "Set fairness deviation threshold", 0., 0.9, 0.2, 0.05)

    # Compute fairness measures
    privi_info = PROTECTED_FEATURES[protected_attribute]
    aif_metric = get_aif_metric(
        valid_fai,
        y_valid,
        y_pred,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    alg_fai(aif_metric, threshold, fairness_metrics=METRICS_TO_USE)

    with st.expander("Notes"):
        fairness_notes()


def compare():
    protected_attribute = st.selectbox(
        "Select protected column.", list(PROTECTED_FEATURES.keys()))

    # Load data
    valid = load_data("output/valid.csv")
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    # Get predictions
    orig_y_pred, orig_text_model_perf = prepare_pred(
        x_valid, y_valid, debias=False)
    y_pred, text_model_perf = prepare_pred(
        x_valid, y_valid, debias=True)

    st.header("Model Performance")
    st.subheader("Before Mitigation")
    st.text(orig_text_model_perf)
    st.subheader("After Mitigation")
    st.text(text_model_perf)

    st.header("Algorithmic Fairness Metrics")
    threshold = st.slider(
        "Set fairness deviation threshold", 0., 0.9, 0.2, 0.05)
    lower = 1 - threshold
    upper = 1 / lower
    st.write(
        "Model is considered fair for the metric when "
        f"**ratio is between {lower:.2f} and {upper:.2f}**."
    )

    # Compute fairness measures
    privi_info = PROTECTED_FEATURES[protected_attribute]
    orig_aif_metric = get_aif_metric(
        valid,
        y_valid,
        orig_y_pred,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    orig_fmeasures = compute_fairness_measures(orig_aif_metric)
    orig_fmeasures["Fair?"] = orig_fmeasures["Ratio"].apply(
        lambda x: "Yes" if lower < x < upper else "No")

    aif_metric = get_aif_metric(
        valid,
        y_valid,
        y_pred,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    fmeasures = compute_fairness_measures(aif_metric)
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if lower < x < upper else "No")

    for m in METRICS_TO_USE:
        source = pd.concat([orig_fmeasures.query(f"Metric == '{m}'"),
                            fmeasures.query(f"Metric == '{m}'")])
        source["Metric"] = ["1-Before Mitigation", "2-After Mitigation"]

        st.write(m)
        st.altair_chart(
            fmeasures_chart(source, lower, upper), use_container_width=True)


if __name__ == "__main__":
    fai()
