import streamlit as st

from xai_fairness.app_fai import fai, compare


def main():
    select = st.sidebar.selectbox(
        "Select dashboard",
        ["Fairness Before Mitigation", "Fairness After Mitigation", "Comparison"])
    
    if select == "Fairness Before Mitigation":
        st.title("Fairness Before Mitigation")
        fai(debias=False)
    elif select == "Fairness After Mitigation":
        st.title("Fairness After Mitigation")
        fai(debias=True)
    elif select == "Comparison":
        st.title("Comparison Before and After Mitigation")
        compare()

    
if __name__ == "__main__":
    main()
