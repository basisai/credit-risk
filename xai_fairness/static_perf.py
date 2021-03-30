"""
Helpers for metrics
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

from xai_fairness.toolkit_perf import (
    cumulative_gain_curve, binary_ks_curve)


def confusion_matrix_chart(source, title="Confusion matrix"):
    """Confusion matrix."""
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


def roc_chart(fpr, tpr, title="ROC curve"):
    """ROC curve."""
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


def line_chart(source, xtitle, ytitle, title):
    """General line chart."""
    base = alt.Chart(source).properties(title=title)
    line = base.mark_line().encode(
        alt.X("x:Q", title=xtitle),
        alt.Y("y:Q", title=ytitle),
        tooltip=[alt.Tooltip("x", title=xtitle), alt.Tooltip("y", title=ytitle)],
    )
    return line


def precision_recall_chart(precision, recall, title="Precision-Recall curve"):
    """PR curve."""
    source = pd.DataFrame({"x": recall, "y": precision})
    return line_chart(source, "Recall", "Precision", title=title)


def cumulative_gain_chart(percentages, gains, title="Cumulative gain curve"):
    """Cumulative gain curve."""
    source = pd.DataFrame({"x": percentages, "y": gains})
    return line_chart(source, "Percentage of samples selected",
                      "Percentage of positive labels", title=title)


def cumulative_lift_chart(percentages, gains, title="Cumulative lift curve"):
    """Cumulative lift curve."""
    source = pd.DataFrame({"x": percentages, "y": gains})
    return line_chart(source, "Percentage of samples selected", "Lift", title=title)


def recall_k_chart(percentages, recall, title="Recall@K"):
    """Recall@k curve."""
    source = pd.DataFrame({"x": percentages, "y": recall})
    return line_chart(source, "Percentage of samples selected", "Recall", title=title)


def precision_k_chart(percentages, precision, title="Precision@K"):
    """Precision@k curve."""
    source = pd.DataFrame({"x": percentages, "y": precision})
    return line_chart(source, "Percentage of samples selected", "Precision", title=title)


def ks_statistic_chart(thresholds, pct0, pct1, max_dist_at, title="KS statistic curve"):
    """KS statistic curve."""
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
    st.altair_chart(
        confusion_matrix_chart(cm).properties(width=300, height=270),
        use_container_width=False,
    )

    fpr, tpr, _ = metrics.roc_curve(actual, proba)
    st.altair_chart(
        roc_chart(fpr, tpr),
        use_container_width=False,
    )

    precision, recall, _ = metrics.precision_recall_curve(actual, proba)
    st.altair_chart(
        precision_recall_chart(precision, recall),
        use_container_width=False,
    )

    percentages, gains = cumulative_gain_curve(actual, proba)
    st.altair_chart(
        cumulative_gain_chart(percentages, gains),
        use_container_width=False,
    )

    idx = (percentages != 0)
    percentages, gains = cumulative_lift_curve(percentages[idx], gains[idx] / percentages[idx])
    st.altair_chart(
        cumulative_lift_chart(percentages, gains),
        use_container_width=False,
    )

    # st.altair_chart(
    #     recall_k_chart(percentages, recall),
    #     use_container_width=False,
    # )

    # st.altair_chart(
    #     precision_k_chart(percentages, precision),
    #     use_container_width=False,
    # )

    thresholds, pct0, pct1, ks_stat, max_dist_at, _ = binary_ks_curve(
        actual, proba)
    st.altair_chart(
        ks_statistic_chart(
            thresholds, pct0, pct1, max_dist_at,
            title=f"KS statistic = {ks_stat:.4f}"),
        use_container_width=False,
    )


def scatter_chart(source, xtitle, ytitle, title):
    """General scatter chart."""
    base = alt.Chart(source).properties(title=title)
    scatter = base.mark_circle(size=60).encode(
        alt.X("x:Q", title=xtitle),
        alt.Y("y:Q", title=ytitle),
        tooltip=[alt.Tooltip("x", title=xtitle), alt.Tooltip("y", title=ytitle)],
    )
    return scatter


def residual_predicted_chart(predicted, residuals, title="Residual vs Predicted Values"):
    """Residual vs predicted values curve."""
    source = pd.DataFrame({"x": predicted, "y": residuals})
    return scatter_chart(source, "Prediction", "Residual", title=title)



def predicted_actual_chart(actual, predicted, title="Predicted vs Actual Values"):
    """Predicted vs actual values curve."""
    source = pd.DataFrame({"x": actual, "y": predicted})
    scatter = scatter_chart(source, "Actual", "Residual", title=title)

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
    residuals = predicted - actual
    st.altair_chart(
        plot_residual_predicted(predicted, residuals),
        use_container_width=False,
    )
    st.altair_chart(
        plot_predicted_actual(actual, predicted),
        use_container_width=False,
    )
