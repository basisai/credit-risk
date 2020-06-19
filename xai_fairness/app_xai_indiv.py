import shap
import streamlit as st

from app_utils import load_model, load_data
from preprocess.constants import FEATURES, TARGET
from .static_xai import make_source_waterfall, waterfall_chart


@st.cache
def get_sk_ids(series):
    return series.tolist()


def xai_indiv():
    clf = load_model("output/lgb_model.pkl")
    sample = load_data("output/test.gz.parquet", num_rows=100)
    sk_ids = get_sk_ids(sample["SK_ID_CURR"])

    # Load explainer
    explainer = shap.TreeExplainer(clf)
    
    # Select instance
    sk_id = st.selectbox("Select SK_ID", sk_ids, 0)
    instance = sample.query(f"SK_ID_CURR == '{sk_id}'")
    x_instance = instance[FEATURES]
    y_instance = instance[TARGET].item()

    st.subheader("Feature values")
    st.dataframe(x_instance.T)

    st.subheader("Actual label")
    st.write(y_instance)
    
    st.subheader("Prediction")
    st.text(clf.predict_proba(x_instance)[0])
    
    # Compute SHAP values
    st.subheader("SHAP values")
    shap_values = explainer.shap_values(x_instance)[1][0]
    base_value = explainer.expected_value[1]
    source = make_source_waterfall(x_instance, base_value, shap_values, max_display=20)
    st.altair_chart(waterfall_chart(source).properties(height=500),
                    use_container_width=True)
    

if __name__ == "__main__":
    xai_indiv()
