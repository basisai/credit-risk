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


def plot_confusion_matrix_by_group(aif_metric, figsize=(16, 4)):
    """Plot confusion matrix by group."""
    def _format_aif360_to_sklearn(aif360_mat):
        return np.array([[aif360_mat['TN'], aif360_mat['FP']],
                         [aif360_mat['FN'], aif360_mat['TP']]])

    cmap = plt.get_cmap('Blues')
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    axs[0].set_title('all')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=None))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[0])
    axs[0].set_xlabel('predicted values')
    axs[0].set_ylabel('actual values')

    axs[1].set_title('privileged')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=True))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[1])
    axs[1].set_xlabel('predicted values')
    axs[1].set_ylabel('actual values')

    axs[2].set_title('unprivileged')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=False))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[2])
    axs[2].set_xlabel('predicted values')
    axs[2].set_ylabel('actual values')
    return fig


def plot_performance_by_group(aif_metric, metric_name, ax=None):
    """Plot performance by group."""
    def _add_annotations(ax):
        for p in ax.patches:
            ax.annotate(format(p.get_height(), ".3f"),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center",
                        va="center",
                        xytext=(0, -10),
                        textcoords="offset points")

    df = get_perf_measure_by_group(aif_metric, metric_name)

    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(x="Group", y=metric_name, data=df, ax=ax)
    ax.set_title("{} by group".format(metric_name))
    ax.set_xlabel(None)

    _add_annotations(ax)
    return ax


def get_fairness(
        grdtruth,
        predicted,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        threshold=0.2,
    ):
    """Fairness wrapper function."""
    clf_metric = ClassificationMetric(
        grdtruth,
        predicted,
        unprivileged_groups=[{protected_attribute: v} for v in unprivileged_attribute_values],
        privileged_groups=[{protected_attribute: v} for v in privileged_attribute_values],
    )
    fmeasures = compute_fairness_measures(clf_metric)
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if np.abs(x - 1) < threshold else "No")

    print(f"Fairness is when deviation from 1 is less than {threshold}")
    display(fmeasures.iloc[:3].style.applymap(color_red, subset=["Fair?"]))

    fig_confmats = plot_confusion_matrix_by_group(clf_metric)

    fig_perfs, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i, metric_name in enumerate([
            'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC',
            'selection_rate', 'precision', 'recall', 'sensitivity',
            'specificity', 'power', 'error_rate']):
        plot_performance_by_group(clf_metric, metric_name, ax=axs[i // 4][i % 4])

    return fmeasures, fig_confmats, fig_perfs
