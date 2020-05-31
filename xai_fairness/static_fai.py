import numpy as np
import altair as alt
import streamlit as st
from aif360.metrics.classification_metric import ClassificationMetric

from .toolkit import (
    prepare_dataset,
    compute_fairness_metrics,
    get_perf_measure_by_group,
    plot_confusion_matrix_by_group,
)


def get_fmeasures(x_val,
                  y_val,
                  y_pred,
                  protected_attribute,
                  privileged_attribute_values,
                  unprivileged_attribute_values,
                  fthresh=0.2,
                  fairness_metrics=None):
    grdtruth = prepare_dataset(
        x_val,
        y_val,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
    )

    predicted = prepare_dataset(
        x_val,
        y_pred,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
    )

    model_metric = ClassificationMetric(
        grdtruth,
        predicted,
        unprivileged_groups=[{protected_attribute: v} for v in unprivileged_attribute_values],
        privileged_groups=[{protected_attribute: v} for v in privileged_attribute_values],
    )

    fmeasures = compute_fairness_metrics(model_metric)
    if fairness_metrics is not None:
        fmeasures = fmeasures[fmeasures["Metric"].isin(fairness_metrics)]
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if np.abs(x - 1) < fthresh else "No")
    return fmeasures, model_metric


def plot_hist(source):
    var = source.columns[0]
    base = alt.Chart(source)
    chart = base.mark_area(
        opacity=0.5, interpolate="step",
    ).encode(
        alt.X("Prediction:Q", bin=alt.Bin(maxbins=20), title="Prediction"),
        alt.Y("count()", stack=None),
        alt.Color(f"{var}:N"),
    )
    mean = base.mark_rule().encode(
        alt.X("mean(Prediction):Q"),
        alt.Color(f"{var}:N"),
        size=alt.value(2),
    )
    return chart + mean


def plot_fmeasures_bar(df0, threshold, mode=None):
    df = df0.copy()
    if mode == "deviation":
        df["Deviation"] = df["Ratio"] - 1
        df["min_val"] = -threshold
        df["max_val"] = threshold
    else:
        df["min_val"] = 1 - threshold
        df["max_val"] = 1 + threshold
    
    base = alt.Chart(df)
    if mode == "deviation":
        bar = base.mark_bar().encode(
            alt.X("Deviation:Q", scale=alt.Scale(domain=[-1., 1.])),
            color=alt.condition(f"abs(datum.Deviation) > {threshold}",
                                alt.value("#FF0D57"), alt.value("#1E88E5")),
            y="Metric:O",
            tooltip=["Metric", "Ratio", "Deviation"],
        )
        rule1 = base.mark_rule(color="red").encode(
            alt.X("min_val:Q"),
            size=alt.value(2),
        )
        rule2 = base.mark_rule(color="red").encode(
            alt.X("max_val:Q", title="Deviation"),
            size=alt.value(2),
        )
    else:
        bar = base.mark_bar().encode(
            alt.X("Ratio:Q"),
            color=alt.condition(f"abs(datum.Ratio - 1) > {threshold}",
                                alt.value("#FF0D57"), alt.value("#1E88E5")),
            y="Metric:O",
            tooltip=["Metric", "Ratio"],
        )
        rule1 = base.mark_rule(color="black").encode(
            alt.X("min_val:Q"),
            size=alt.value(2),
        )
        rule2 = base.mark_rule(color="black").encode(
            alt.X("max_val:Q", title="Ratio"),
            size=alt.value(2),
        )
    return bar + rule1 + rule2


def color_red(x):
    return "color: red" if x == "No" else "color: black"


def alg_fai(fmeasures, model_metric, fthresh):
    st.subheader("Fairness Metrics")
    st.write(f"Fairness is when **ratio is between {1-fthresh:.2f} and {1+fthresh:.2f}**.")
    
    chart = plot_fmeasures_bar(fmeasures, fthresh)
    st.altair_chart(chart, use_container_width=True)
    
    st.dataframe(
        fmeasures[["Metric", "Unprivileged", "Privileged", "Ratio", "Fair?"]]
        .style.applymap(color_red, subset=["Fair?"])
    )
    
    st.subheader("Performance Metrics")
    all_perfs = []
    for metric_name in [
            'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC',
            'selection_rate', 'precision', 'recall', 'sensitivity',
            'specificity', 'power', 'error_rate']:
        df = get_perf_measure_by_group(model_metric, metric_name)
        c = alt.Chart(df).mark_bar().encode(
            x=f"{metric_name}:Q",
            y="Group:O",
            tooltip=["Group", metric_name],
        )
        all_perfs.append(c)
    
    all_charts = alt.concat(*all_perfs, columns=1)
    st.altair_chart(all_charts, use_container_width=False)
    
    st.subheader("Confusion Matrices")
    st.pyplot(plot_confusion_matrix_by_group(model_metric), figsize=(8, 6))
