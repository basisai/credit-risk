"""
Streamlit app.
"""
import streamlit as st

from dashboard.app_fai import fai
from dashboard.app_xai_indiv import xai_indiv
from dashboard.app_analysis import compare_models


def main():
    select = st.sidebar.selectbox("Select dashboard", [
        "Fairness",
        "Individual XAI",
        "Credit Risk Analysis",
    ])
    st.title(select)

    if select == "Fairness":
        fai(debias=False)
    elif select == "Individual XAI":
        xai_indiv()
    elif select == "Credit Risk Analysis":
        compare_models()

    
if __name__ == "__main__":
    main()
