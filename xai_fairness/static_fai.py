"""
Helpers for fairness
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from .toolkit import (
    get_aif_metric,
    compute_fairness_measures,
    get_perf_measure_by_group,
    color_red,
)


def plot_hist(source, cutoff):
    """Plot custom histogram."""
    source["Cutoff"] = cutoff
    var = source.columns[0]
    base = alt.Chart(source)
    chart = base.mark_area(
        opacity=0.5, interpolate="step",
    ).encode(
        alt.X("Prediction:Q", bin=alt.Bin(maxbins=20), title="Prediction"),
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
        alt.Y("Metric:O"),
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


def custom_fmeasures(aif_metric, threshold=0.2, fairness_metrics=None):
    """To customise fairness measures dataframe."""
    fmeasures = compute_fairness_measures(aif_metric)
    if fairness_metrics is not None:
        fmeasures = fmeasures[fmeasures["Metric"].isin(fairness_metrics)]
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if np.abs(x - 1) < threshold else "No")
    return fmeasures


def alg_fai_summary(valid, unq_fai_classes, true_class, pred_class, config_fai, config):
    """Fairness summary."""
    threshold = config["fairness_threshold"]
    st.write("Algorithmic fairness assesses the models based on technical definitions of fairness. "
             "If all are met, the model is deemed to be fair.")
    st.write(f"Fairness deviation threshold is set at **{threshold}**. "
             "Absolute fairness is 1, so a model is considered fair for the metric when the "
             f"**metric is between {1 - threshold:.2f} and {1 + threshold:.2f}**.")

    final_fairness = []
    for attr, attr_values in config_fai.items():
        st.subheader(f"Prohibited Feature: `{attr}`")

        for fcl in unq_fai_classes:
            _true_class = (true_class == fcl).astype(int)
            _pred_class = (pred_class == fcl).astype(int)

            # Compute fairness measures
            aif_metric = get_aif_metric(
                valid,
                _true_class,
                _pred_class,
                attr,
                attr_values["privileged_attribute_values"],
                attr_values["unprivileged_attribute_values"],
            )
            fmeasures = custom_fmeasures(
                aif_metric,
                threshold=config["fairness_threshold"],
                fairness_metrics=config["fairness_metrics"],
            )

            if len(unq_fai_classes) > 2:
                st.write(f"**Fairness Class `{attr}={fcl}` vs rest**")
            st.dataframe(
                fmeasures[["Metric", "Ratio", "Fair?"]]
                .style.applymap(color_red, subset=["Fair?"])
            )
            st.altair_chart(plot_fmeasures_bar(fmeasures, config["fairness_threshold"]),
                            use_container_width=True)
            if np.mean(fmeasures["Fair?"] == "Yes") > 0.6:
                st.write("Overall: **Fair**")
                final_fairness.append([f"{attr}-class{fcl}", "Yes"])
            else:
                st.write("Overall: **Not Fair**")
                final_fairness.append([f"{attr}-class{fcl}", "No"])

    final_fairness = pd.DataFrame(final_fairness, columns=["Prohibited Variable", "Fair?"])
    return final_fairness


def get_confusion_matrix_chart(cm, title):
    """Plot custom confusion matrix chart."""
    source = pd.DataFrame([[0, 0, cm['TN']],
                           [0, 1, cm['FP']],
                           [1, 0, cm['FN']],
                           [1, 1, cm['TP']],
                           ], columns=["actual values", "predicted values", "count"])

    base = alt.Chart(source).encode(
        y='actual values:O',
        x='predicted values:O',
    ).properties(
        width=200,
        height=200,
        title=title,
    )
    rects = base.mark_rect().encode(
        color='count:Q',
    )
    text = base.mark_text(
        align='center',
        baseline='middle',
        color='black',
        size=12,
        dx=0,
    ).encode(
        text='count:Q',
    )
    return rects + text


def alg_fai(fmeasures, aif_metric, threshold):
    st.write(f"Fairness is when **ratio is between {1-threshold:.2f} and {1+threshold:.2f}**.")

    chart = plot_fmeasures_bar(fmeasures, threshold)
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        fmeasures[["Metric", "Unprivileged", "Privileged", "Ratio", "Fair?"]]
        .style.applymap(color_red, subset=["Fair?"])
    )

    st.write("**Performance Metrics**")
    all_perfs = []
    for metric_name in [
            'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC',
            'selection_rate', 'precision', 'recall', 'sensitivity',
            'specificity', 'power', 'error_rate']:
        df = get_perf_measure_by_group(aif_metric, metric_name)
        c = alt.Chart(df).mark_bar().encode(
            x=f"{metric_name}:Q",
            y="Group:O",
            tooltip=["Group", metric_name],
        )
        all_perfs.append(c)

    all_charts = alt.concat(*all_perfs, columns=1)
    st.altair_chart(all_charts, use_container_width=False)

    st.write("**Confusion Matrices**")
    cm1 = aif_metric.binary_confusion_matrix(privileged=None)
    c1 = get_confusion_matrix_chart(cm1, "All")
    st.altair_chart(alt.concat(c1, columns=2), use_container_width=False)
    cm2 = aif_metric.binary_confusion_matrix(privileged=True)
    c2 = get_confusion_matrix_chart(cm2, "Privileged")
    cm3 = aif_metric.binary_confusion_matrix(privileged=False)
    c3 = get_confusion_matrix_chart(cm3, "Unprivileged")
    st.altair_chart(c2 | c3, use_container_width=False)


def alg_fai_appendix(valid, unq_fai_classes, true_class, pred_class, config_fai, config):
    for attr, attr_values in config_fai.items():
        st.subheader(f"Prohibited Feature: `{attr}`")
        for fcl in unq_fai_classes:
            _true_class = (true_class == fcl).astype(int)
            _pred_class = (pred_class == fcl).astype(int)

            # Compute fairness measures
            aif_metric = get_aif_metric(
                valid,
                _true_class,
                _pred_class,
                attr,
                attr_values["privileged_attribute_values"],
                attr_values["unprivileged_attribute_values"],
            )
            fmeasures = custom_fmeasures(
                aif_metric,
                threshold=config["fairness_threshold"],
                fairness_metrics=config["fairness_metrics"],
            )

            if len(unq_fai_classes) > 2:
                st.write(f"**Fairness Class `{attr}={fcl}` vs rest**")
            alg_fai(fmeasures, aif_metric, config["fairness_threshold"])
