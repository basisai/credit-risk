import base64
from pathlib import Path

import streamlit as st

from xai_fairness.app_fai import fai
from xai_fairness.app_xai_indiv import xai_indiv
from credit_analysis.analysis import compare_models


def uri_encode_path(path, mime="image/png"):
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def add_header(path):
    st.markdown(
        "<img src='{}' class='img-fluid'>".format(uri_encode_path(path)),
        unsafe_allow_html=True,
    )


def main():
    # max_width = 1000
    # st.markdown(
    #     f"""
    #     <style>
    #     .reportview-container .main .block-container{{
    #         max-width: {max_width}px;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )
    add_header("assets/logo.png")

    select = st.sidebar.selectbox(
        "Select dashboard",
        ["Fairness", "Individual Instance Explainability", "Credit Risk Analysis"])
    st.title(select)

    if select == "Fairness":
        fai(debias=False)
    elif select == "Individual Instance Explainability":
        xai_indiv()
    elif select == "Credit Risk Analysis":
        compare_models()

    
if __name__ == "__main__":
    main()
