"""
Helpers for fairness
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from xai_fairness.toolkit_fai import (
    compute_fairness_measures,
    get_perf_measure_by_group,
)


def binarize(y, label):
    """Binarize array-like data according to label."""
    return (np.array(y) == label).astype(int)


def color_red(x):
    """Styling: color red."""
    return "color: red" if x == "No" else "color: black"


def plot_hist(source, cutoff):
    """Plot custom histogram."""
    source["Cutoff"] = cutoff
    var = source.columns[0]
    base = alt.Chart(source)
    chart = base.mark_area(
        opacity=0.5, interpolate="step",
    ).encode(
        alt.X("Prediction:Q", bin=alt.Bin(maxbins=10), title="Prediction"),
        alt.Y("count()", stack=None),
        alt.Color(f"{var}:N"),
    )
    rule = base.mark_rule(color="red").encode(
        alt.X("Cutoff:Q"),
        size=alt.value(2),
    )
    mean = base.mark_rule().encode(
        alt.X("mean(Prediction):Q"),
        alt.Color(f"{var}:N"),
        size=alt.value(2),
    )
    return chart + rule + mean


def plot_fmeasures_bar(df, threshold):
    """Plot custom bar chart."""
    source = df.copy()
    source["lbd"] = 1 - threshold
    source["ubd"] = 1 + threshold

    base = alt.Chart(source)
    bars = base.mark_bar().encode(
        alt.X("Ratio:Q"),
        alt.Y("Metric:O", sort=alt.SortField("order")),
        alt.Color("Fair?:N", scale=alt.Scale(
            domain=["Yes", "No"], range=["#1E88E5", "#FF0D57"])),
        alt.Tooltip(["Metric", "Ratio"]),
    )
    rule1 = base.mark_rule(color="black").encode(
        alt.X("lbd:Q"),
        size=alt.value(2),
    )
    rule2 = base.mark_rule(color="black").encode(
        alt.X("ubd:Q", title="Ratio"),
        size=alt.value(2),
    )
    return bars + rule1 + rule2


def plot_confusion_matrix(cm, title):
    """Plot custom confusion matrix."""
    source = pd.DataFrame(
        [
            ["negative", "negative", cm["TN"]],
            ["negative", "positive", cm["FP"]],
            ["positive", "negative", cm["FN"]],
            ["positive", "positive", cm["TP"]],
        ],
        columns=["actual", "predicted", "value"],
    )

    base = alt.Chart(source).encode(
        y="actual:O",
        x="predicted:O",
    ).properties(
        width=200,
        height=200,
        title=title,
    )
    rects = base.mark_rect().encode(
        color="value:Q",
    )
    text = base.mark_text(
        align="center",
        baseline="middle",
        color="black",
        size=12,
        dx=0,
    ).encode(
        text="value:Q",
    )
    return rects + text


def custom_fmeasures(aif_metric, threshold=0.2, fairness_metrics=None):
    """To customise fairness measures dataframe."""
    fmeasures = compute_fairness_measures(aif_metric)
    if fairness_metrics is not None:
        fmeasures = fmeasures.query(f"Metric == {fairness_metrics}")
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if np.abs(x - 1) < threshold else "No")
    return fmeasures


def alg_fai(fmeasures, aif_metric, threshold):
    st.write(f"Fairness is when **ratio is between {1 - threshold:.2f} and {1 + threshold:.2f}**.")

    chart = plot_fmeasures_bar(fmeasures, threshold)
    st.altair_chart(chart, use_container_width=True)

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


def fairness_notes():
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")
    st.write("**Predictive equality**:")
    st.latex(r"\frac{\text{FPR}(D=\text{unprivileged})}{\text{FPR}(D=\text{privileged})}")
    st.write("**Equalized odds**:")
    st.latex(r"\frac{\text{TPR}(D=\text{unprivileged})}{\text{TPR}(D=\text{privileged})} \text{ and } \frac{\text{FPR}(D=\text{unprivileged})}{\text{FPR}(D=\text{privileged})}")
    st.write("**Predictive parity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")
    st.write("**Conditional use accuracy equality**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})} \text{ and } \frac{\text{NPV}(D=\text{unprivileged})}{\text{NPV}(D=\text{privileged})}")
