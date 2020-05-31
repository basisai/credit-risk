import pandas as pd
import altair as alt
import streamlit as st
from sklearn import metrics

from app_utils import load_model, load_data, predict
from preprocess.constants import FEATURES, TARGET, CONFIG_FAI
from .static_fai import get_fmeasures, plot_fmeasures_bar, color_red
from .toolkit import prepare_dataset, get_perf_measure_by_group

METRICS_TO_USE = ["Equal opportunity", "Predictive parity", "Statistical parity"]


def get_confusion_matrix_chart(cm, title):
    source = pd.DataFrame([[0, 0, cm['TN']],
                           [0, 1, cm['FP']],
                           [1, 0, cm['FN']],
                           [1, 1, cm['TP']],
                           ], columns=["actual values", "predicted values", "count"])

    base = alt.Chart(source).encode(
        y='actual values:O',
        x='predicted values:O',
    ).properties(
        width=200,
        height=200,
        title=title,
    )
    rects = base.mark_rect().encode(
        color='count:Q',
    )
    text = base.mark_text(
        align='center',
        baseline='middle',
        color='black',
        size=12,
        dx=0,
    ).encode(
        text='count:Q',
    )
    # chart = alt.layer(rects, text).configure_axis(
    #     labelFontSize=12,
    #     titleFontSize=12,
    #     domainWidth=0.0,
    # )
    return rects + text


def print_model_perf(y_val, y_pred):
    text = ""
    text += "Model accuracy = {:.4f}\n".format(metrics.accuracy_score(y_val, y_pred))
    text += "Weighted Average Precision = {:.4f}\n".format(metrics.precision_score(y_val, y_pred, average="weighted"))
    text += "Weighted Average Recall = {:.4f}\n\n".format(metrics.recall_score(y_val, y_pred, average="weighted"))
    text += metrics.classification_report(y_val, y_pred, digits=4)
    return text


@st.cache
def prepare_pred(x_val, y_val, debias=False):
    # Load model
    clf = load_model("output/lgb.pkl")

    # Predict on val data
    y_prob = predict(clf, x_val)

    # st.header("Prediction Distributions")
    cutoff = 0.5  # st.slider("Set probability cutoff", 0., 1., 0.5, 0.01, key="proba")
    y_pred = (y_prob > cutoff).astype(int)

    if debias:
        model = load_model("output/eq_odds_sex.pkl")
        attr = "Sex=Male"
        predicted_val = prepare_dataset(
            x_val, y_pred, attr,
            CONFIG_FAI[attr]["privileged_attribute_values"],
            CONFIG_FAI[attr]["unprivileged_attribute_values"])

        adj_pred_val = model.predict(predicted_val)
        y_pred = adj_pred_val.labels

    # Model performance
    text_model_perf = print_model_perf(y_val, y_pred)

    return y_pred, text_model_perf


def fai(debias=False):
    # st.subheader("User Write-up")
    # if debias:
    #     st.write(CONFIG["after_mitigation"])
    # else:
    #     st.write(CONFIG["before_mitigation"])

    protected_attribute = st.selectbox("Select protected column.", list(CONFIG_FAI.keys()))
    privileged_attribute_values = CONFIG_FAI[protected_attribute]["privileged_attribute_values"]
    unprivileged_attribute_values = CONFIG_FAI[protected_attribute]["unprivileged_attribute_values"]

    # Load data
    val = load_data("output/test.gz.parquet").fillna(0)  # Fairness does not allow NaNs
    x_val = val[FEATURES]
    y_val = val[TARGET].values

    # Get predictions
    y_pred, text_model_perf = prepare_pred(x_val, y_val, debias=debias)

    # source = pd.DataFrame({
    #     select_protected: x_val[select_protected].values,
    #     "Predicted Probability": y_prob,
    # })
    # source["Cutoff"] = cutoff
    # hist = plot_hist(source)
    # st.altair_chart(hist, use_container_width=True)
    
    st.header("Model Performance")
    st.text(text_model_perf)

    st.header("Algorithmic Fairness Metrics")
    fthresh = st.slider("Set fairness deviation threshold", 0., 1., 0.2, 0.05)
    st.write(f"Absolute fairness is 1. The model is considered fair if **ratio is between {1-fthresh:.2f} and {1+fthresh:.2f}**.")

    # Compute fairness metrics
    fmeasures, clf_metric = get_fmeasures(x_val,
                                          y_val,
                                          y_pred,
                                          protected_attribute,
                                          privileged_attribute_values,
                                          unprivileged_attribute_values,
                                          fthresh=fthresh,
                                          fairness_metrics=METRICS_TO_USE)

    st.altair_chart(plot_fmeasures_bar(fmeasures, fthresh), use_container_width=True)
    
    st.dataframe(
        fmeasures[["Metric", "Unprivileged", "Privileged", "Ratio", "Fair?"]]
        .style.applymap(color_red, subset=["Fair?"])
        .format({"Unprivileged": "{:.3f}", "Privileged": "{:.3f}", "Ratio": "{:.3f}"})
    )

    st.subheader("Confusion Matrices")
    # st.pyplot(plot_confusion_matrix_by_group(clf_metric), figsize=(8, 6))
    cm1 = clf_metric.binary_confusion_matrix(privileged=None)
    c1 = get_confusion_matrix_chart(cm1, "All")
    st.altair_chart(alt.concat(c1, columns=2), use_container_width=False)
    cm2 = clf_metric.binary_confusion_matrix(privileged=True)
    c2 = get_confusion_matrix_chart(cm2, "Privileged")
    cm3 = clf_metric.binary_confusion_matrix(privileged=False)
    c3 = get_confusion_matrix_chart(cm3, "Unprivileged")
    st.altair_chart(c2 | c3, use_container_width=False)

    st.header("Annex")
    st.subheader("Performance Metrics")
    all_perfs = []
    for metric_name in [
            'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC',
            'selection_rate', 'precision', 'recall', 'sensitivity',
            'specificity', 'power', 'error_rate']:
        df = get_perf_measure_by_group(clf_metric, metric_name)
        c = alt.Chart(df).mark_bar().encode(
            x=f"{metric_name}:Q",
            y="Group:O",
            tooltip=["Group", metric_name],
        )
        all_perfs.append(c)
    
    all_charts = alt.concat(*all_perfs, columns=1)
    st.altair_chart(all_charts, use_container_width=False)

    st.subheader("Notes")
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    st.write("**Predictive parity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")


def chart_cm_comparison(orig_clf_metric, clf_metric, privileged, title):
    cm1 = orig_clf_metric.binary_confusion_matrix(privileged=privileged)
    cm2 = clf_metric.binary_confusion_matrix(privileged=privileged)
    c1 = get_confusion_matrix_chart(cm1, f"{title}: Before Mitigation")
    c2 = get_confusion_matrix_chart(cm2, f"{title}: After Mitigation")
    return c1 | c2


def compare():
    protected_attribute = st.selectbox("Select protected column.", list(CONFIG_FAI.keys()))
    privileged_attribute_values = CONFIG_FAI[protected_attribute]["privileged_attribute_values"]
    unprivileged_attribute_values = CONFIG_FAI[protected_attribute]["unprivileged_attribute_values"]

    # Load data
    val = load_data("output/test.gz.parquet").fillna(0)  # Fairness does not allow NaNs
    x_val = val[FEATURES]
    y_val = val[TARGET].values

    # Get predictions
    orig_y_pred, orig_text_model_perf = prepare_pred(x_val, y_val, debias=False)
    y_pred, text_model_perf = prepare_pred(x_val, y_val, debias=True)

    st.header("Model Performance")
    st.subheader("Before Mitigation")
    st.text(orig_text_model_perf)
    st.subheader("After Mitigation")
    st.text(text_model_perf)

    st.header("Algorithmic Fairness Metrics")
    fthresh = st.slider("Set fairness deviation threshold", 0., 1., 0.2, 0.05)
    st.write(
        f"Absolute fairness is 1. The model is considered fair if **ratio is between {1 - fthresh:.2f} and {1 + fthresh:.2f}**.")

    # Compute fairness metrics
    orig_fmeasures, orig_clf_metric = get_fmeasures(
        x_val, y_val, orig_y_pred, protected_attribute,
        privileged_attribute_values, unprivileged_attribute_values,
        fthresh, METRICS_TO_USE,
    )
    fmeasures, clf_metric = get_fmeasures(
        x_val, y_val, y_pred, protected_attribute,
        privileged_attribute_values, unprivileged_attribute_values,
        fthresh, METRICS_TO_USE,
    )

    for m in METRICS_TO_USE:
        source = pd.concat([orig_fmeasures.query(f"Metric == '{m}'"),
                            fmeasures.query(f"Metric == '{m}'")])
        source["Metric"] = ["1-Before Mitigation", "2-After Mitigation"]

        st.write(m)
        st.altair_chart(plot_fmeasures_bar(source, fthresh), use_container_width=True)

#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, None, "All"),
#                     use_container_width=False)
#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, True, "Privileged"),
#                     use_container_width=False)
#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, False, "Unprivileged"),
#                     use_container_width=False)

    
if __name__ == "__main__":
    fai()
