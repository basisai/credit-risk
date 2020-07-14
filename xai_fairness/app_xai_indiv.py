import altair as alt
import streamlit as st

from app_utils import load_model, load_data, predict, compute_shap
from preprocess.constants import FEATURES, TARGET
from .static_xai import make_source_waterfall, waterfall_chart

TARGET_CLASSES = [0, 1]


@st.cache
def get_sk_ids(series):
    return series.tolist()


def plot_hist(source):
    """Plot custom histogram."""
    base = alt.Chart(source)
    chart = base.mark_area(
        opacity=0.5, interpolate="step",
    ).encode(
        alt.X("Prediction:Q", bin=alt.Bin(maxbins=10), title="Prediction"),
        alt.Y("count()", stack=None),
    )
    return chart


def xai_indiv():
    clf = load_model("output/lgb_model.pkl")
    sample = load_data("output/test.gz.parquet", num_rows=100)
    # sk_ids = get_sk_ids(sample["SK_ID_CURR"])
    x_sample = sample[FEATURES]
    y_sample = sample[TARGET].values

    scores = predict(clf, x_sample)

    all_shap_values, all_base_value = compute_shap(clf, x_sample)

    idx = 0
    if TARGET_CLASSES is not None and len(TARGET_CLASSES) > 2:
        select_class = st.selectbox("Select class", TARGET_CLASSES, 1)
        idx = TARGET_CLASSES.index(select_class)

    # TODO
    score_df = sample[[TARGET]].copy()
    score_df["Prediction"] = scores
    charts = [plot_hist(score_df[score_df[TARGET] == lb]).properties(title=f"Class = {lb}")
              for lb in TARGET_CLASSES]
    st.altair_chart(alt.concat(*charts, columns=2), use_container_width=True)

    # customized
    bin_options = [f"{i / 10:.1f} - {(i + 1) / 10:.1f}" for i in range(10)]
    scores_bin = (scores * 10).astype(int)

    select_bin = st.selectbox("Select prediction bin", bin_options)
    bin_idx = bin_options.index(select_bin)
    select_class = st.selectbox("Select class", TARGET_CLASSES, 1)
    class_idx = TARGET_CLASSES.index(select_class)
    select_samples = sample.index[(y_sample == class_idx) & (scores_bin == bin_idx)]
    
    # Select instance
    _row_idx = st.slider("Select instance", 0, len(select_samples), 0)
    row_idx = select_samples[_row_idx]
    instance = x_sample.iloc[row_idx: row_idx + 1]

    st.write("**Feature values**")
    st.dataframe(instance.T)

    st.write(f"**Actual label: `{y_sample[row_idx]}`**")
    st.write(f"**Prediction: `{scores[row_idx]:.4f}`**")
    
    # Compute SHAP values
    st.subheader("Feature SHAP contribution to prediction")
    shap_values = all_shap_values[idx][row_idx]
    base_value = all_base_value[idx]
    source = make_source_waterfall(instance, base_value, shap_values, max_display=15)
    st.altair_chart(waterfall_chart(source).properties(height=500),
                    use_container_width=True)
    

if __name__ == "__main__":
    xai_indiv()
