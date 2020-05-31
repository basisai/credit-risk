import streamlit as st

from xai_fairness.app_fai import fai
from xai_fairness.app_xai_indiv import xai_indiv

def main():
    # max_width = 1000  #st.sidebar.slider("Set page width", min_value=700, max_value=1500, value=1000, step=20)
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

    select = st.sidebar.selectbox(
        "Select dashboard",
        ["Fairness", "Individual Instance Explainability"])
    
    if select == "Fairness":
        st.title("Fairness")
        fai(debias=False)
    elif select == "Individual Instance Explainability":
        st.title("Individual Instance Explainability")
        xai_indiv()

    
if __name__ == "__main__":
    main()
