"""
Toolkit for XAI.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def get_explainer(
        model=None,
        model_type=None,
        predict_func=None,
        bkgrd_data=None,
        kmeans_size=10,
    ):
    """Function to select the SHAP explainer.
    Use the relevant explainer for each type of model.
    :param Optional model: Model to compute shap values for. In case model is of unsupported
        type, use predict_func to pass in a generic function instead
    :param Optional[str] model_type: Type of the model
    :param Optional[Callable] predict_func: Generic function to compute shap values for.
        It should take a matrix of samples (# samples x # features) and compute the
        output of the model for those samples.
        The output can be a vector (# samples) or a matrix (# samples x # model outputs).
    :param: Optional[pandas.DataFrame] bkgrd_data: background data for explainability analysis
    :param: Optional[int] kmeans_size: Number of k-means clusters.
        Only required for explaining generic predict_func
    :return explainer
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    else:
        if bkgrd_data is None:
            raise ValueError("Non tree model requires background data")
        if model_type == "linear":
            explainer = shap.LinearExplainer(model, bkgrd_data)
        else:
            explainer = _get_kernel_explainer(predict_func, bkgrd_data, kmeans_size)
    return explainer


def _get_kernel_explainer(predict_func, bkgrd_data, kmeans_size=10):
    if predict_func is None:
        raise ValueError("No target to compute shap values. Expected either model or predict_func")
    # rather than use the whole training set to estimate expected values,
    # summarize with a set of weighted kmeans, each weighted by
    # the number of points they represent.
    if kmeans_size is None:
        x_bkgrd_summary = bkgrd_data
    else:
        x_bkgrd_summary = shap.kmeans(bkgrd_data, kmeans_size)
    return shap.KernelExplainer(predict_func, x_bkgrd_summary)


def compute_shap(explainer, x):
    """Get shap_values and base_value."""
    all_shap = explainer.shap_values(x)
    all_base = np.array(explainer.expected_value).reshape(-1)

    if len(all_base) == 1:
        # regressor or binary XGBClassifier
        return [all_shap], all_base

    if len(all_base) == 2:
        # binary classifier, only take the values for class=1
        return all_shap[1:], all_base[1:]

    # multiclass classifier
    return all_shap, all_base


def compute_corrcoef(features, shap_values):
    """
    Compute correlation between each feature and its SHAP values.
    :param pandas.DataFrame features:
    :param numpy.array shap_values:
    :return numpy.array: (shape = (dim of predict output, number of features))
    """
    all_corrs = list()
    for cls_shap_val in shap_values:
        corrs = list()
        for i in range(features.shape[1]):
            df_ = pd.DataFrame({"x": features.iloc[:, i].values, "y": cls_shap_val[:, i]})
            corrs.append(df_.corr(method="pearson").values[0, 1])
        all_corrs.append(np.array(corrs))
    return all_corrs


def shap_summary_plot(
        shap_values,
        features,
        feature_names=None,
        max_display=None,
        plot_size=(12, 6),
        show=False,
    ):
    """Plot SHAP summary plot."""
    # TODO: convert to altair chart
    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        max_display=max_display,
        plot_size=plot_size,
        show=show,
    )
    plt.tight_layout()
    return fig


def shap_dependence_plot(
        ind,
        shap_values=None,
        features=None,
        feature_names=None,
        interaction_index="auto",
        show=False,
        plot_size=(12, 6),
    ):
    """Plot dependence interaction chart."""
    # TODO: convert to altair chart
    fig, ax = plt.subplots(figsize=plot_size)
    shap.dependence_plot(
        ind,
        shap_values=shap_values,
        features=features,
        feature_names=feature_names,
        interaction_index=interaction_index,
        ax=ax,
        show=show,
    )
    plt.tight_layout()
    return fig


def pdp_plot(
        model,
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
        show_percentile=False,
    ):
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


def actual_plot(
        model,
        X,
        feature,
        feature_name,
        num_grid_points=10,
        xticklabels=None,
        show_percentile=False,
    ):
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


def target_plot(
        df,
        feature,
        feature_name,
        target,
        num_grid_points=10,
        xticklabels=None,
        show_percentile=False,
    ):
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


def pdp_interact_plot(
        model,
        dataset,
        model_features,
        feature1,
        feature2,
        plot_type="grid",
        x_quantile=True,
        plot_pdp=False,
    ):
    """Wrapper for pdp.pdp_interact_plot. Uses pdp.pdp_interact."""
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=dataset,
        model_features=model_features,
        features=[feature1, feature2],
    )

    fig, _ = pdp.pdp_interact_plot(
        pdp_interact_out=pdp_interact_out,
        feature_names=[feature1, feature2],
        plot_type=plot_type,
        x_quantile=x_quantile,
        plot_pdp=plot_pdp,
    )
    return fig
