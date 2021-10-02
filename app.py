"""
Streamlit app.
"""
import streamlit as st

from dashboard.app_fai import fai
from dashboard.app_xai_indiv import xai_indiv
from dashboard.app_analysis import compare_models, analyse_model


def main():
    st.sidebar.title("Credit Risk")
    select = st.sidebar.radio("Select dashboard", [
        "Fairness",
        "Individual Explainability",
        "Credit Risk Analysis",
    ])
    st.title(select)

    if select == "Fairness":
        fai(debias=False)
    elif select == "Individual Explainability":
        xai_indiv()
    elif select == "Credit Risk Analysis":
        compare_models()

    
if __name__ == "__main__":
    main()
