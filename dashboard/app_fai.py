"""
App for FAI.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from xai_fairness.toolkit_fai import (
    get_aif_metric,
    get_perf_measure_by_group,
)
from xai_fairness.static_fai import (
    compute_fairness_measures,
    plot_confusion_matrix,
    plot_fmeasures_bar,
    color_red,
)

from data.utils import load_model, load_data, predict, print_model_perf
from preprocess.constants import FEATURES, TARGET, CONFIG_FAI

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
    protected_attribute = st.selectbox("Select protected column.", list(CONFIG_FAI.keys()))

    # Load data
    valid = load_data("data/test.gz.parquet").fillna(0)
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values
    valid_fai = valid[list(CONFIG_FAI.keys())]

    # Get predictions
    y_pred, text_model_perf = prepare_pred(x_valid, y_valid, debias=debias)

    st.header("Model Performance")
    st.text(text_model_perf)

    st.header("Algorithmic Fairness Metrics")
    fthresh = st.slider("Set fairness deviation threshold", 0., 1., 0.2, 0.05)
    st.write("Absolute fairness is 1. The model is considered fair "
             f"if **ratio is between {1-fthresh:.2f} and {1+fthresh:.2f}**.")

    # Compute fairness measures
    privi_info = CONFIG_FAI[protected_attribute]
    aif_metric = get_aif_metric(
        valid_fai,
        y_valid,
        y_pred,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    fmeasures = compute_fairness_measures(aif_metric)
    fmeasures = fmeasures[fmeasures["Metric"].isin(METRICS_TO_USE)]
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if np.abs(x - 1) < fthresh else "No")

    st.altair_chart(plot_fmeasures_bar(fmeasures, fthresh), use_container_width=True)
    
    st.dataframe(
        fmeasures[["Metric", "Unprivileged", "Privileged", "Ratio", "Fair?"]]
        .style.applymap(color_red, subset=["Fair?"])
        .format({"Unprivileged": "{:.3f}", "Privileged": "{:.3f}", "Ratio": "{:.3f}"})
    )

    st.subheader("Confusion Matrices")
    cm1 = aif_metric.binary_confusion_matrix(privileged=None)
    c1 = plot_confusion_matrix(cm1, "All")
    st.altair_chart(alt.concat(c1, columns=2), use_container_width=False)
    cm2 = aif_metric.binary_confusion_matrix(privileged=True)
    c2 = plot_confusion_matrix(cm2, "Privileged")
    cm3 = aif_metric.binary_confusion_matrix(privileged=False)
    c3 = plot_confusion_matrix(cm3, "Unprivileged")
    st.altair_chart(c2 | c3, use_container_width=False)

    st.header("Annex")
    st.subheader("Performance Metrics")
    all_perfs = []
    for metric_name in [
            "TPR", "TNR", "FPR", "FNR", "PPV", "NPV", "FDR", "FOR", "ACC",
            "selection_rate", "precision", "recall", "sensitivity",
            "specificity", "power", "error_rate"]:
        df = get_perf_measure_by_group(aif_metric, metric_name)
        c = alt.Chart(df).mark_bar().encode(
            x=f"{metric_name}:Q",
            y="Group:O",
            tooltip=["Group", metric_name],
        )
        all_perfs.append(c)
    
    all_charts = alt.concat(*all_perfs, columns=1)
    st.altair_chart(all_charts, use_container_width=False)

    st.subheader("Notes")
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    st.write("**Predictive parity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")


def chart_cm_comparison(orig_clf_metric, clf_metric, privileged, title):
    cm1 = orig_clf_metric.binary_confusion_matrix(privileged=privileged)
    cm2 = clf_metric.binary_confusion_matrix(privileged=privileged)
    c1 = get_confusion_matrix_chart(cm1, f"{title}: Before Mitigation")
    c2 = get_confusion_matrix_chart(cm2, f"{title}: After Mitigation")
    return c1 | c2


def compare():
    protected_attribute = st.selectbox("Select protected column.", list(CONFIG_FAI.keys()))

    # Load data
    valid = load_data("output/valid.csv")
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    # Get predictions
    orig_y_pred, orig_text_model_perf = prepare_pred(x_valid, y_valid, debias=False)
    y_pred, text_model_perf = prepare_pred(x_valid, y_valid, debias=True)

    st.header("Model Performance")
    st.subheader("Before Mitigation")
    st.text(orig_text_model_perf)
    st.subheader("After Mitigation")
    st.text(text_model_perf)

    st.header("Algorithmic Fairness Metrics")
    fthresh = st.slider("Set fairness deviation threshold", 0., 1., 0.2, 0.05)
    st.write("Absolute fairness is 1. The model is considered fair "
             f"if **ratio is between {1 - fthresh:.2f} and {1 + fthresh:.2f}**.")

    # Compute fairness measures
    privi_info = CONFIG_FAI[protected_attribute]
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
        lambda x: "Yes" if np.abs(x - 1) < fthresh else "No")

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
        lambda x: "Yes" if np.abs(x - 1) < fthresh else "No")

    for m in METRICS_TO_USE:
        source = pd.concat([orig_fmeasures.query(f"Metric == '{m}'"),
                            fmeasures.query(f"Metric == '{m}'")])
        source["Metric"] = ["1-Before Mitigation", "2-After Mitigation"]

        st.write(m)
        st.altair_chart(plot_fmeasures_bar(source, fthresh), use_container_width=True)

    
if __name__ == "__main__":
    fai()
