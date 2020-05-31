"""
Script containing commonly used functions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.metrics.classification_metric import ClassificationMetric
from pdpbox import pdp, info_plots
from IPython.display import display

plt.style.use('seaborn-darkgrid')


def pdp_plot(model,
             dataset,
             model_features,
             feature,
             feature_name,
             num_grid_points=10,
             xticklabels=None,
             plot_lines=False,
             frac_to_plot=1,
             plot_pts_dist=False,
             x_quantile=False,
             show_percentile=False):
    """Wrapper for pdp.pdp_plot. Uses pdp.pdp_isolate."""
    pdp_iso = pdp.pdp_isolate(
        model=model,
        dataset=dataset,
        model_features=model_features,
        feature=feature,
        num_grid_points=num_grid_points,
    )
    
    fig, axes = pdp.pdp_plot(
        pdp_iso,
        feature_name,
        plot_lines=plot_lines,
        frac_to_plot=frac_to_plot,
        plot_pts_dist=plot_pts_dist,
        x_quantile=x_quantile,
        show_percentile=show_percentile,
    )
    
    if xticklabels is not None:
        if plot_lines:
            _ = axes["pdp_ax"]["_count_ax"].set_xticklabels(xticklabels)
        else:
            _ = axes["pdp_ax"].set_xticklabels(xticklabels)
    return fig


def actual_plot(model,
                X,
                feature,
                feature_name,
                num_grid_points=10,
                xticklabels=None,
                show_percentile=False):
    """Wrapper for info_plots.actual_plot."""
    fig, axes, summary_df = info_plots.actual_plot(
        model=model,
        X=X,
        feature=feature,
        feature_name=feature_name,
        num_grid_points=num_grid_points,
        show_percentile=show_percentile,
        predict_kwds={},
    )
    
    if xticklabels is not None:
        _ = axes["bar_ax"].set_xticklabels(xticklabels)
    return fig, summary_df


def target_plot(df,
                feature,
                feature_name,
                target,
                num_grid_points=10,
                xticklabels=None,
                show_percentile=False):
    """Wrapper for info_plots.target_plot."""
    fig, axes, summary_df = info_plots.target_plot(
        df=df,
        feature=feature,
        feature_name=feature_name,
        target=target,
        num_grid_points=num_grid_points,
        show_percentile=show_percentile,
    )
    
    if xticklabels is not None:
        _ = axes["bar_ax"].set_xticklabels(xticklabels)
    return fig, summary_df


def pdp_interact_plot(model,
                      dataset,
                      model_features,
                      feature1,
                      feature2,
                      plot_type="grid",
                      x_quantile=True,
                      plot_pdp=False):
    """Wrapper for pdp.pdp_interact_plot. Uses pdp.pdp_interact."""
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=dataset,
        model_features=model_features,
        features=[feature1, feature2],
    )
    
    fig, axes = pdp.pdp_interact_plot(
        pdp_interact_out=pdp_interact_out,
        feature_names=[feature1, feature2],
        plot_type=plot_type,
        x_quantile=x_quantile,
        plot_pdp=plot_pdp,
    )
    return fig


def prepare_dataset(features,
                    labels,
                    protected_attribute,
                    privileged_attribute_values,
                    unprivileged_attribute_values):
    """Prepare dataset for computing fairness metrics."""
    df = features.copy()
    df['outcome'] = labels
    
    return BinaryLabelDataset(
        df=df,
        label_names=['outcome'],
        scores_names=[],
        protected_attribute_names=[protected_attribute],
        privileged_protected_attributes=[np.array(privileged_attribute_values)],
        unprivileged_protected_attributes=[np.array(unprivileged_attribute_values)],
    )


def get_fairness(grdtruth,
                 predicted,
                 protected_attribute,
                 privileged_attribute_values,
                 unprivileged_attribute_values,
                 threshold=0.2):
    """Fairness wrapper function."""
    clf_metric = ClassificationMetric(
        grdtruth,
        predicted,
        unprivileged_groups=[{protected_attribute: v} for v in unprivileged_attribute_values],
        privileged_groups=[{protected_attribute: v} for v in privileged_attribute_values],
    )
    fmeasures = compute_fairness_metrics(clf_metric)
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(lambda x: "Yes" if np.abs(x - 1) < threshold else "No")
    
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


def compute_fairness_metrics(aif_metric):
    """Compute and report fairness metrics."""
    fmeasures = []
    
    # Equal opportunity: equal FNR
    fnr_ratio = aif_metric.false_negative_rate_ratio()
    fmeasures.append([
        "Equal opportunity",
        "Separation",
        aif_metric.false_negative_rate(),
        aif_metric.false_negative_rate(False),
        aif_metric.false_negative_rate(True),
        fnr_ratio,
    ])
    
    # Predictive parity: equal PPV
    ppv_all = aif_metric.positive_predictive_value()
    ppv_up = aif_metric.positive_predictive_value(False)
    ppv_p = aif_metric.positive_predictive_value(True)
    ppv_ratio = ppv_up / ppv_p
    fmeasures.append([
        "Predictive parity",
        "Sufficiency",
        ppv_all,
        ppv_up,
        ppv_p,
        ppv_ratio,
    ])

    # Statistical parity
    disparate_impact = aif_metric.disparate_impact()
    fmeasures.append([
        "Statistical parity",
        "Independence",
        aif_metric.selection_rate(),
        aif_metric.selection_rate(False),
        aif_metric.selection_rate(True),
        disparate_impact,
    ])
    
    # Predictive equality: equal FPR
    fpr_ratio = aif_metric.false_positive_rate_ratio()
    fmeasures.append([
        "Predictive equality",
        "Separation",
        aif_metric.false_positive_rate(),
        aif_metric.false_positive_rate(False),
        aif_metric.false_positive_rate(True),
        fpr_ratio,
    ])

    # Equalized odds: equal TPR and equal FPR
    eqodds_all = (aif_metric.true_positive_rate() + aif_metric.false_positive_rate()) / 2
    eqodds_up = (aif_metric.true_positive_rate(False) + aif_metric.false_positive_rate(False)) / 2
    eqodds_p = (aif_metric.true_positive_rate(True) + aif_metric.false_positive_rate(True)) / 2
    eqodds_ratio = eqodds_up / eqodds_p
    fmeasures.append([
        "Equalized odd",
        "Separation",
        eqodds_all,
        eqodds_up,
        eqodds_p,
        eqodds_ratio,
    ])
    
    # Conditional use accuracy equality: equal PPV and equal NPV
    acceq_all = (aif_metric.positive_predictive_value(False) + aif_metric.negative_predictive_value(False)) / 2
    acceq_up = (aif_metric.positive_predictive_value(False) + aif_metric.negative_predictive_value(False)) / 2
    acceq_p = (aif_metric.positive_predictive_value(True) + aif_metric.negative_predictive_value(True)) / 2
    acceq_ratio = acceq_up / acceq_p
    fmeasures.append([
        "Conditional use accuracy equality",
        "Sufficiency",
        acceq_all,
        acceq_up,
        acceq_p,
        acceq_ratio,
    ])

    return pd.DataFrame(fmeasures, columns=[
        "Metric", "Criterion", "All", "Unprivileged", "Privileged", "Ratio"])


def plot_confusion_matrix_by_group(aif_metric, figsize=(16, 4)):
    """Plot confusion matrix by group."""
    def _format_aif360_to_sklearn(aif360_mat):
        return np.array([[aif360_mat['TN'], aif360_mat['FP']],
                         [aif360_mat['FN'], aif360_mat['TP']]])

    cmap = plt.get_cmap('Blues')
    fig, axs = plt.subplots(1,3, figsize=figsize)

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


def get_perf_measure_by_group(aif_metric, metric_name):
    perf_measures = ['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC']
    
    func_dict = {
        'selection_rate': lambda x: aif_metric.selection_rate(privileged=x),
        'precision': lambda x: aif_metric.precision(privileged=x),
        'recall': lambda x: aif_metric.recall(privileged=x),
        'sensitivity': lambda x: aif_metric.sensitivity(privileged=x),
        'specificity': lambda x: aif_metric.specificity(privileged=x),
        'power': lambda x: aif_metric.power(privileged=x),
        'error_rate': lambda x: aif_metric.error_rate(privileged=x),
    }
    
    if metric_name in perf_measures:
        metric_func = lambda x: aif_metric.performance_measures(privileged=x)[metric_name]  
    elif metric_name in func_dict.keys():
        metric_func = func_dict[metric_name]
    else:
        raise NotImplementedError

    df = pd.DataFrame({
        'Group': ['all', 'privileged', 'unprivileged'],
        metric_name: [metric_func(group) for group in [None, True, False]],
    })
    return df


def plot_performance_by_group(aif_metric, metric_name, ax=None):
    """Plot performance by group."""
    def _add_annotations(ax):
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.3f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center',
                        xytext = (0, -10), textcoords = 'offset points')
        
    df = get_perf_measure_by_group(aif_metric, metric_name)

    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(x='Group', y=metric_name, data=df, ax=ax)
    ax.set_title('{} by group'.format(metric_name))
    ax.set_xlabel(None)
    
    _add_annotations(ax)

    
def color_red(x):
    return "color: red" if x == "No" else "color: black"
