"""
Helpers for XAI
"""
import numpy as np
import pandas as pd
import shap
import altair as alt
import streamlit as st
from pdpbox import pdp


@st.cache(allow_output_mutation=True)
def compute_pdp_isolate(model, dataset, model_features, feature):
    pdp_isolate_out = pdp.pdp_isolate(
        model=model,
        dataset=dataset,
        model_features=model_features,
        feature=feature,
        num_grid_points=15,
    )
    return pdp_isolate_out


def pdp_chart(pdp_isolate_out, feature_name):
    """Plot pdp charts."""
    source = pd.DataFrame({
        "feature": pdp_isolate_out.feature_grids,
        "value": pdp_isolate_out.pdp,
    })

    if pdp_isolate_out.feature_type == "numeric":
        chart = alt.Chart(source).mark_line().encode(
            x=alt.X("feature", title=feature_name),
            y=alt.Y("value", title=""),
            tooltip=["feature", "value"],
        )
    else:
        source["feature"] = source["feature"].astype(str)
        chart = alt.Chart(source).mark_bar().encode(
            x=alt.X("value", title=""),
            y=alt.Y("feature", title=feature_name, sort="-x"),
            tooltip=["feature", "value"],
        )
    return chart


@st.cache(allow_output_mutation=True)
def compute_pdp_interact(model, dataset, model_features, features):
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=dataset,
        model_features=model_features,
        features=features,
    )
    return pdp_interact_out


def pdp_heatmap(pdp_interact_out, feature_names):
    """Plot pdp heatmap."""
    source = pdp_interact_out.pdp

    for i in [0, 1]:
        if pdp_interact_out.feature_types[i] == "onehot":
            value_vars = pdp_interact_out.feature_grids[i]
            id_vars = list(set(source.columns) - set(value_vars))
            source = pd.melt(source, value_vars=value_vars,
                             id_vars=id_vars, var_name=feature_names[i])
            source = source[source["value"] == 1].drop(columns=["value"])

        elif pdp_interact_out.feature_types[i] == "binary":
            source[feature_names[i]] = source[feature_names[i]].astype(str)

    chart = alt.Chart(source).mark_rect().encode(
        x=feature_names[0],
        y=feature_names[1],
        color="preds",
        tooltip=feature_names + ["preds"]
    )
    return chart


def _convert_name(ind, feature_names):
    """Get index of feature name if it is given."""
    if isinstance(ind, str):
        return np.where(np.array(feature_names) == ind)[0][0]
    return ind


def make_source_dp(shap_values, features, feature_names, feature):
    ind = _convert_name(feature, feature_names)

    # randomize the ordering so plotting overlaps are not related to data ordering
    oinds = np.arange(shap_values.shape[0])
    np.random.shuffle(oinds)

    return pd.DataFrame({
        feature: features[oinds, ind],
        "value": shap_values[oinds, ind],
    })


def _is_numeric(series, max_unique=16):
    """Flag if series is numeric."""
    if len(set(series.values[:3000])) > max_unique:
        return True
    return False


def dependence_chart(source, feat_col, val_col="value"):
    if _is_numeric(source[feat_col]):
        scatterplot = alt.Chart(source).mark_circle(size=8).encode(
            x=alt.X(f"{feat_col}:Q"),
            y=alt.Y(f"{val_col}:Q", title="SHAP value"),
        )
        return scatterplot

    stripplot = alt.Chart(source, width=40).mark_circle(size=8).encode(
        x=alt.X(
            "jitter:Q",
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y(f"{val_col}:Q", title="SHAP value"),
        color=alt.Color(f"{feat_col}:N", legend=None),
        column=alt.Column(
            f"{feat_col}:N",
            header=alt.Header(
                labelAngle=-90,
                titleOrient="top",
                labelOrient="bottom",
                labelAlign="right",
                labelPadding=3,
            ),
        ),
    ).transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    return stripplot
