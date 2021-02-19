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
        color='preds',
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
            'jitter:Q',
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y(f'{val_col}:Q', title="SHAP value"),
        color=alt.Color(f'{feat_col}:N', legend=None),
        column=alt.Column(
            f'{feat_col}:N',
            header=alt.Header(
                labelAngle=-90,
                titleOrient='top',
                labelOrient='bottom',
                labelAlign='right',
                labelPadding=3,
            ),
        ),
    ).transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    return stripplot


def make_source_waterfall(instance, base_value, shap_values, max_display=10):
    df = pd.melt(instance)
    df.columns = ["feature", "feature_value"]
    df["shap_value"] = shap_values

    df["val_"] = df["shap_value"].abs()
    df = df.sort_values("val_", ascending=False)

    df["val_"] = df["shap_value"].values
    remaining = df["shap_value"].iloc[max_display:].sum()
    output_value = df["shap_value"].sum() + base_value

    _df = df.iloc[:max_display]

    df0 = pd.DataFrame({"feature": ["Average Model Output"],
                        "shap_value": [base_value],
                        "val_": [base_value]})
    df1 = _df.query("shap_value > 0").sort_values("shap_value", ascending=False).copy()
    df2 = pd.DataFrame({"feature": ["Others"],
                        "shap_value": [remaining],
                        "val_": [remaining]})
    df3 = _df.query("shap_value < 0").sort_values("shap_value").copy()
    df4 = pd.DataFrame({"feature": ["Individual Observation"],
                        "shap_value": [output_value],
                        "val_": [0]})
    source = pd.concat([df0, df1, df2, df3, df4], axis=0, ignore_index=True)

    source["close"] = source["val_"].cumsum()
    source["open"] = source["close"].shift(1)
    source.loc[len(source) - 1, "open"] = 0
    source["open"].fillna(0, inplace=True)
    return source


def waterfall_chart(source, decimal=3):
    """Plot waterfall chart."""
    source = source.copy()
    for c in ["feature_value", "shap_value"]:
        source[c] = source[c].round(decimal).astype(str)
    source.loc[source["feature_value"] == "nan", "feature_value"] = ""

    bars = alt.Chart(source).mark_bar().encode(
        alt.X("feature:O", sort=source["feature"].tolist(), axis=alt.Axis(labelLimit=120)),
        alt.Y("open:Q", scale=alt.Scale(zero=False), title=""),
        alt.Y2("close:Q"),
        alt.Tooltip(["feature", "feature_value", "shap_value"]),
    )
    color1 = bars.encode(
        color=alt.condition(
            "datum.open <= datum.close",
            alt.value("#FF0D57"),
            alt.value("#1E88E5"),
        ),
    )
    color2 = bars.encode(
        color=alt.condition(
            "datum.feature == 'Average Model Output' || datum.feature == 'Individual Observation'",
            alt.value("#F7E0B6"),
            alt.value(""),
        ),
    )
    text = bars.mark_text(
        align='center',
        baseline='middle',
        dy=-5,
        color='black',
    ).encode(
        text='feature_value:N',
    )
    return bars + color1 + color2 + text


def plot_hist(source):
    """Plot custom histogram."""
    base = alt.Chart(source)
    chart = base.mark_area(
        opacity=0.5, interpolate="step",
    ).encode(
        alt.X("Prediction:Q", bin=alt.Bin(maxbins=10), title="Prediction"),
        alt.Y("count()", stack=None),
    ).properties(
        width=280,
        height=200,
    )
    return chart


###############################################################################
# Additional
def xai_charts(corr_df, shap_values, x_valid, feature_names, max_display, max_rows=3000):
    """Plot global XAI charts."""
    st.write("**SHAP Summary Plots of Top Features**")

    source = corr_df.iloc[:max_display].copy()
    source["corr"] = source["corrcoef"].apply(lambda x: "positive" if x > 0 else "negative")
    chart = alt.Chart(source).mark_bar().encode(
        alt.X("mas_value:Q", title="mean(|SHAP value|) (average impact on model output magnitude)"),
        alt.Y("feature:N", title="", sort="-x"),
        alt.Color("corr:N", scale=alt.Scale(
            domain=["positive", "negative"], range=["#FF0D57", "#1E88E5"])),
        alt.Tooltip(["feature", "mas_value"]),
    )
    st.altair_chart(chart, use_container_width=True)

    fig = shap_summary_plot(
        shap_values[:max_rows],
        x_valid.iloc[:max_rows],
        feature_names=feature_names,
        max_display=max_display,
        plot_size=[12, 6],
        show=False,
    )
    st.pyplot(fig)


def model_xai_summary(shap_summary_dfs, all_shap_values, x_valid, feature_names, config, is_multiclass):
    overall_top_df = pd.DataFrame({"feature": feature_names})
    for lb, df in enumerate(shap_summary_dfs):
        if is_multiclass:
            st.subheader(f"Target Class `{lb}`")

        top_df = shap_summary_dfs[lb].sort_values("mas_value", ascending=False, ignore_index=True)
        overall_top_df = pd.merge(overall_top_df, top_df[["feature", "mas_value"]], on="feature")
        xai_charts(top_df,
                   all_shap_values[lb],
                   x_valid,
                   feature_names,
                   config["num_top_features"])

    if is_multiclass:
        overall_top_df["total_val"] = overall_top_df.iloc[:, 1:].sum(axis=1)
        overall_top_df.sort_values("total_val", ascending=False, inplace=True, ignore_index=True)
        top_features = overall_top_df["feature"].iloc[:config["num_top_features"]].tolist()
        return top_features, None

    top_df = top_df.iloc[:config["num_top_features"]]
    top_features = top_df["feature"].tolist()
    st.write("The top features are `" + "`, `".join(top_features[:5]) + "`.")
    dict_feats = {
        "pos": top_df.query("corrcoef > 0")["feature"].values,
        "neg": top_df.query("corrcoef < 0")["feature"].values,
    }
    return top_features, dict_feats


def model_xai_appendix(all_shap_values, x_valid, feature_names, top_features, is_multiclass):
    features = x_valid.values

    for feat_name in top_features:
        st.subheader(f"Feature: `{feat_name}`")
        title = ""
        for lb, shap_values in enumerate(all_shap_values):
            if is_multiclass:
                title = f"Target Class {lb}"

            source = make_source_dp(shap_values, features, feature_names, feat_name)
            st.altair_chart(
                dependence_chart(source, feat_name).properties(title=title),
                use_container_width=False,
            )


def indiv_xai(instance, base_value, shap_values, title="", max_display=10):
    source = make_source_waterfall(instance, base_value, shap_values, max_display=max_display)
    chart = waterfall_chart(source).properties(
        title=title,
    )
    st.altair_chart(chart, use_container_width=True)


def indiv_xai_appendix(indiv_samples, indiv_shap_values, indiv_base_values, config, is_multiclass):
    for i, (tcl, x) in enumerate(indiv_samples.items()):
        for lb, (shap_values, base_value) in enumerate(zip(indiv_shap_values, indiv_base_values)):
            title = f"Sample from Class={tcl}: SHAP Contribution to Model Prediction"
            if is_multiclass:
                title += f" for Class={lb}"

            st.write(f"**{title}**")
            indiv_xai(x,
                      base_value,
                      shap_values[i],
                      max_display=config["num_top_features"])
