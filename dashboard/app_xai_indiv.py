"""
App for individual XAI.
"""
import altair as alt
import streamlit as st

from xai_fairness.static_xai_indiv import (
    make_source_waterfall,
    waterfall_chart,
    histogram_chart,
)

from data.utils import load_model, load_data, predict
from preprocess.constants import FEATURES, TARGET, TARGET_CLASSES


def xai_indiv():
    clf = load_model("data/lgb_model.pkl")
    shap_df = load_data("data/shap_df.gz.parquet")
    sample = load_data("data/test.gz.parquet")
    x_sample = sample[FEATURES]
    y_sample = sample[TARGET].values
    scores = predict(clf, x_sample)

    # TODO
    score_df = sample[[TARGET]].copy()
    score_df["Prediction"] = scores
    charts = [histogram_chart(score_df[score_df[TARGET] == lb]).properties(title=f"Class = {lb}")
              for lb in TARGET_CLASSES]
    st.altair_chart(alt.concat(*charts, columns=2), use_container_width=True)

    # customized
    bin_options = [f"{i / 10:.1f} - {(i + 1) / 10:.1f}" for i in range(10)]
    scores_bin = (scores * 10).astype(int)

    c0, c1 = st.beta_columns(2)
    select_class = c0.selectbox("Select class", TARGET_CLASSES, 1)
    class_idx = TARGET_CLASSES.index(select_class)
    select_bin = c1.selectbox("Select prediction bin", bin_options)
    bin_idx = bin_options.index(select_bin)
    select_samples = x_sample.index[(y_sample == class_idx) & (scores_bin == bin_idx)]
    
    if len(select_samples) == 0:
        st.write("**No instances found.**")
        return

    # Select instance
    _row_idx = st.slider("Select instance", 0, len(select_samples), 0)
    row_idx = select_samples[_row_idx]
    instance = x_sample.iloc[row_idx: row_idx + 1]

    st.write(f"**Actual label: `{y_sample[row_idx]}`**")
    st.write(f"**Prediction: `{scores[row_idx]:.4f}`**")

    # Compute SHAP values
    st.subheader("Feature SHAP contribution to prediction")
    shap_values = shap_df[FEATURES].iloc[row_idx].values
    base_value = shap_df["base_value"].iloc[row_idx]
    source = make_source_waterfall(instance, base_value, shap_values, max_display=15)
    st.altair_chart(waterfall_chart(source), use_container_width=True)
    
    df = instance.copy().T
    df.columns = ["feature_value"]
    df["shap_value"] = shap_values
    st.write(df)


if __name__ == "__main__":
    xai_indiv()
