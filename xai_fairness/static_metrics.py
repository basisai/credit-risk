"""
Helpers for metrics
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

from xai_fairness.toolkit_metrics import (
    cumulative_gain_curve, cumulative_lift_curve, binary_ks_curve)


def plot_confusion_matrix(source, title="Confusion matrix"):
    """Plot confusion matrix."""
    base = alt.Chart(source).encode(
        x="predicted:O",
        y="actual:O",
    ).properties(
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


def plot_roc(fpr, tpr, title="ROC curve"):
    """Plot ROC curve."""
    source = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    base = alt.Chart(source).properties(title=title)
    line = base.mark_line(color="red").encode(
        alt.X("FPR", title="False positive rate"),
        alt.Y("TPR", title="True positive rate"),
        alt.Tooltip(["FPR", "TPR"]),
    )
    area = base.mark_area(fillOpacity=0.5, fill="red").encode(
        alt.X("FPR"),
        alt.Y("TPR"),
    )

    _df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    baseline = alt.Chart(_df).mark_line(strokeDash=[20, 5], color="black").encode(
        alt.X("x"),
        alt.Y("y"),
    )
    return line + area + baseline


def _plot_line(source, xtitle, ytitle, title):
    base = alt.Chart(source).properties(title=title)
    line = base.mark_line().encode(
        alt.X("x:Q", title=xtitle),
        alt.Y("y:Q", title=ytitle),
        tooltip=[alt.Tooltip("x", title=xtitle), alt.Tooltip("y", title=ytitle)],
    )
    return line


def plot_precision_recall(precision, recall, title="Precision-Recall curve"):
    """Plot PR curve."""
    source = pd.DataFrame({"x": recall, "y": precision})
    return _plot_line(source, "Recall", "Precision", title=title)


def plot_cumulative_gain(percentages, gains, title="Cumulative gain curve"):
    """Plot cumulative gain curve."""
    source = pd.DataFrame({"x": percentages, "y": gains})
    return _plot_line(source, "Percentage of samples selected",
                      "Percentage of positive labels", title=title)


def plot_cumulative_lift(percentages, gains, title="Cumulative lift curve"):
    """Plot cumulative lift curve."""
    source = pd.DataFrame({"x": percentages, "y": gains})
    return _plot_line(source, "Percentage of samples selected", "Lift", title=title)


def plot_recall_k(percentages, recall, title="Recall@K"):
    """Plot recall@k curve."""
    source = pd.DataFrame({"x": percentages, "y": recall})
    return _plot_line(source, "Percentage of samples selected", "Recall", title=title)


def plot_precision_k(percentages, precision, title="Precision@K"):
    """Plot precision@k curve."""
    source = pd.DataFrame({"x": percentages, "y": precision})
    return _plot_line(source, "Percentage of samples selected", "Precision", title=title)


def plot_ks_statistic(thresholds, pct0, pct1, max_dist_at, title="KS statistic curve"):
    """Plot KS statistic curve."""
    source = (
        pd.DataFrame({"Threshold": thresholds, "Class_0": pct0, "Class_1": pct1})
        .melt(id_vars=["Threshold"], var_name="Class", value_name="Value")
    )
    base = alt.Chart(source).properties(title=title)
    line = base.mark_line(color="red").encode(
        alt.X("Threshold:Q", title="Threshold"),
        alt.Y("Value:Q", title="Percentage below threshold"),
        alt.Color("Class:N"),
        alt.Tooltip(["Threshold", "Class", "Value"]),
    )

    _df = pd.DataFrame({"x": [max_dist_at, max_dist_at], "y": [0, 1]})
    baseline = alt.Chart(_df).mark_line(strokeDash=[20, 5], color="black").encode(
        alt.X("x"),
        alt.Y("y"),
        alt.Tooltip("x"),
    )
    return line + baseline


def classification_summary(actual, proba, predicted):
    """Classification model performance."""
    cm = metrics.confusion_matrix(actual, predicted)
    labels = sorted(list(set(np.unique(actual)).union(set(np.unique(predicted)))))
    cm = pd.DataFrame(cm, columns=labels)
    cm["actual"] = labels
    cm = cm.melt(id_vars=["actual"], var_name="predicted")
    c1 = plot_confusion_matrix(cm).properties(
        width=300, height=270,
    )
    st.altair_chart(
        c1,
        use_container_width=False,
    )

    fpr, tpr, _ = metrics.roc_curve(actual, proba)
    st.altair_chart(
        plot_roc(fpr, tpr),
        use_container_width=False,
    )

    precision, recall, _ = metrics.precision_recall_curve(actual, proba)
    st.altair_chart(
        plot_precision_recall(precision, recall),
        use_container_width=False,
    )

    percentages, gains = cumulative_gain_curve(actual, proba)
    st.altair_chart(
        plot_cumulative_gain(percentages, gains),
        use_container_width=False,
    )

    percentages, gains = cumulative_lift_curve(actual, proba)
    st.altair_chart(
        plot_cumulative_lift(percentages, gains),
        use_container_width=False,
    )

    # st.altair_chart(
    #     plot_recall_k(percentages, recall),
    #     use_container_width=False,
    # )

    # st.altair_chart(
    #     plot_precision_k(percentages, precision),
    #     use_container_width=False,
    # )

    thresholds, pct0, pct1, ks_stat, max_dist_at, _ = binary_ks_curve(
        actual, proba)
    st.altair_chart(
        plot_ks_statistic(
            thresholds, pct0, pct1, max_dist_at,
            title=f"KS statistic = {ks_stat:.4f}"),
        use_container_width=False,
    )


def _plot_scatter(source, xtitle, ytitle, title):
    base = alt.Chart(source).properties(title=title)
    scatter = base.mark_circle(size=60).encode(
        alt.X("x:Q", title=xtitle),
        alt.Y("y:Q", title=ytitle),
        tooltip=[alt.Tooltip("x", title=xtitle), alt.Tooltip("y", title=ytitle)],
    )
    return scatter


def plot_residual_predicted(predicted, residuals, title="Residual vs Predicted Values"):
    """Plot residual vs predicted values curve."""
    source = pd.DataFrame({"x": predicted, "y": residuals})
    return _plot_scatter(source, "Prediction", "Residual", title=title)



def plot_predicted_actual(actual, predicted, title="Predicted vs Actual Values"):
    """Plot predicted vs actual values curve."""
    source = pd.DataFrame({"x": actual, "y": predicted})
    scatter = _plot_scatter(source, "Actual", "Residual", title=title)

    vmin = source.min().min()
    vmax = source.max().max()
    _df = pd.DataFrame({"x": [vmin, vmax], "y": [vmin, vmax]})
    baseline = alt.Chart(_df).mark_line(strokeDash=[20, 5], color="black").encode(
        alt.X("x"),
        alt.Y("y"),
    )
    return scatter + baseline


def regression_summary(actual, predicted):
    """Regression model performance."""
    residuals = np.abs(actual - predicted)
    st.altair_chart(
        plot_residual_predicted(predicted, residuals),
        use_container_width=False,
    )
    st.altair_chart(
        plot_predicted_actual(actual, predicted),
        use_container_width=False,
    )
