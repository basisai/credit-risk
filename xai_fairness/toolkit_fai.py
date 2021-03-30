"""
Toolkit for fairness.
"""
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics.classification_metric import ClassificationMetric


def prepare_dataset(
        features,
        labels,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=1.,
        unfavorable_label=0.,
    ):
    """Prepare dataset for computing fairness metrics."""
    df = features.copy()
    df["outcome"] = labels

    return BinaryLabelDataset(
        df=df,
        label_names=["outcome"],
        scores_names=list(),
        protected_attribute_names=[protected_attribute],
        privileged_protected_attributes=[np.array(privileged_attribute_values)],
        unprivileged_protected_attributes=[np.array(unprivileged_attribute_values)],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )


def get_aif_metric(
        valid,
        true_class,
        pred_class,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=1.,
        unfavorable_label=0.,
    ):
    """Get aif metric wrapper."""
    grdtruth = prepare_dataset(
        valid,
        true_class,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )

    predicted = prepare_dataset(
        valid,
        pred_class,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )

    aif_metric = ClassificationMetric(
        grdtruth,
        predicted,
        unprivileged_groups=[{protected_attribute: v} for v in unprivileged_attribute_values],
        privileged_groups=[{protected_attribute: v} for v in privileged_attribute_values],
    )
    return aif_metric


def compute_fairness_measures(aif_metric):
    """Compute fairness measures."""
    fmeasures = list()

    # Statistical parity
    fmeasures.append([
        "Statistical parity",
        "Independence",
        aif_metric.selection_rate(),
        aif_metric.selection_rate(False),
        aif_metric.selection_rate(True),
        aif_metric.disparate_impact(),
    ])

    # Equal opportunity: equal FNR
    fmeasures.append([
        "Equal opportunity (equal FNR)",
        "Separation",
        aif_metric.false_negative_rate(),
        aif_metric.false_negative_rate(False),
        aif_metric.false_negative_rate(True),
        aif_metric.false_negative_rate_ratio(),
    ])

    # Predictive equality: equal FPR
    fmeasures.append([
        "Predictive equality (equal FPR)",
        "Separation",
        aif_metric.false_positive_rate(),
        aif_metric.false_positive_rate(False),
        aif_metric.false_positive_rate(True),
        aif_metric.false_positive_rate_ratio(),
    ])

    # Equal TPR
    fmeasures.append([
        "Equal TPR",
        "Separation",
        aif_metric.true_positive_rate(),
        aif_metric.true_positive_rate(False),
        aif_metric.true_positive_rate(True),
        aif_metric.true_positive_rate(False) / aif_metric.true_positive_rate(True),
    ])

    # Predictive parity: equal PPV
    fmeasures.append([
        "Predictive parity (equal PPV)",
        "Sufficiency",
        aif_metric.positive_predictive_value(),
        aif_metric.positive_predictive_value(False),
        aif_metric.positive_predictive_value(True),
        aif_metric.positive_predictive_value(False) / aif_metric.positive_predictive_value(True),
    ])

    # Equal NPV
    fmeasures.append([
        "Equal NPV",
        "Sufficiency",
        aif_metric.negative_predictive_value(),
        aif_metric.negative_predictive_value(False),
        aif_metric.negative_predictive_value(True),
        aif_metric.negative_predictive_value(False) / aif_metric.negative_predictive_value(True),
    ])

    df = pd.DataFrame(fmeasures, columns=[
        "Metric", "Criterion", "All", "Unprivileged", "Privileged", "Ratio"])
    df.index.name = "order"
    df.reset_index(inplace=True)
    return df


def get_perf_measure_by_group(aif_metric, metric_name):
    """Get performance measures by group."""
    perf_measures = ["TPR", "TNR", "FPR", "FNR", "PPV", "NPV", "FDR", "FOR", "ACC"]

    func_dict = {
        "selection_rate": lambda x: aif_metric.selection_rate(privileged=x),
        "precision": lambda x: aif_metric.precision(privileged=x),
        "recall": lambda x: aif_metric.recall(privileged=x),
        "sensitivity": lambda x: aif_metric.sensitivity(privileged=x),
        "specificity": lambda x: aif_metric.specificity(privileged=x),
        "power": lambda x: aif_metric.power(privileged=x),
        "error_rate": lambda x: aif_metric.error_rate(privileged=x),
    }

    if metric_name in perf_measures:
        metric_func = lambda x: aif_metric.performance_measures(privileged=x)[metric_name]
    elif metric_name in func_dict.keys():
        metric_func = func_dict[metric_name]
    else:
        raise NotImplementedError

    df = pd.DataFrame({
        "Group": ["all", "privileged", "unprivileged"],
        metric_name: [metric_func(group) for group in [None, True, False]],
    })
    return df


###############################################################################
# Fairness plots
def fairness_summary(aif_metric, threshold=0.2):
    """Fairness charts wrapper function."""
    lower = 1 - threshold
    upper = 1 / lower
    print(f"Model is considered fair for the metric when **ratio is between {lower:.2f} and {upper:.2f}**.")

    fmeasures = compute_fairness_measures(aif_metric)
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if lower < x < upper else "No")

    display(fmeasures.iloc[:3].style.applymap(color_red, subset=["Fair?"]))

    fig_cm = plot_confusion_matrix_by_group(aif_metric)
    return fmeasures, fig_cm


def plot_confusion_matrix_by_group(aif_metric, figsize=(14, 4)):
    """Plot confusion matrix by group."""
    def _cast_cm(x):
        return np.array([
            [x["TN"], x["FP"]],
            [x["FN"], x["TP"]]
        ])

    cmap = plt.get_cmap("Blues")
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for i, (privileged, title) in enumerate(zip(
            [None, True, False], ["all", "privileged", "unprivileged"])):
        cm = _cast_cm(aif_metric.binary_confusion_matrix(privileged=privileged))
        sns.heatmap(cm, cmap=cmap, annot=True, fmt="g", ax=axs[i])
        axs[i].set_xlabel("predicted")
        axs[i].set_ylabel("actual")
        axs[i].set_title(title)
    return fig
