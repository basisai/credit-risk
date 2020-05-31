import shap
import streamlit as st

from app_utils import load_model, load_data
from preprocess.constants import FEATURES, TARGET
from .static_xai import make_source_waterfall, waterfall_chart


def xai_indiv():
    st.title("Individual Instance Explainability")
    
    clf = load_model("output/lgb.pkl")
    sample = load_data("output/test.gz.parquet")

    # Load explainer
    explainer = shap.TreeExplainer(clf)
    
    # Select instance
    sk_id = st.selectbox("Select SK_ID", 0, sample["SK_ID_CURR"].values, 0)
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
    source = make_source_waterfall(instance, base_value, shap_values, max_display=20)
    st.altair_chart(waterfall_chart(source), use_container_width=True)
    

if __name__ == "__main__":
    xai_indiv()
