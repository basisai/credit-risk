"""
App for credit analysis.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from credit_analysis.toolkit import (
    anova_func,
    ks_func,
    roc_func,
    odds_func,
    score_shift_func,
    acct_table,
    swapset,
)
from data.utils import analysis_data

LOSS_PER_BAD_ACCT = 2000
REV_PER_GOOD_ACCT = 100


def cdf_charts(good_prob, bad_prob):
    """Plot CDF chart."""
    source = pd.concat([
        pd.DataFrame({
            "Type": "Good loans",
            "Prob": good_prob,
            "Value": np.arange(len(good_prob)) / len(good_prob),
        }),
        pd.DataFrame({
            "Type": "Bad loans",
            "Prob": bad_prob,
            "Value": np.arange(len(bad_prob)) / len(bad_prob),
        }),
    ], ignore_index=True)
    lines = alt.Chart(source).mark_line().encode(
        x="Prob",
        y="Value",
        color="Type",
        tooltip=["Prob", "Value", "Type"],
    )
    return lines


def roc_chart(fpr, tpr):
    """Plot ROC curve."""
    source = pd.DataFrame({"FPR": fpr, "TPR": tpr, "random": fpr})
    base = alt.Chart(source).encode(
        alt.X("FPR:Q"),
    )
    line1 = base.mark_line(color="black", strokeDash=[1, 1]).encode(
        alt.Y("random:Q"),
    )
    line2 = base.mark_line().encode(
        alt.Y("TPR:Q", title="TPR"),
    )
    return line1 + line2


def heatmap_chart(df, title=""):
    """Plot custom confusion matrix chart."""
    source = df.copy()
    source = source.reset_index()
    source = pd.melt(source, id_vars="index", value_vars=df.columns)
    source.columns = ["m1", "m2", "value"]

    base = alt.Chart(source).encode(
        alt.X("m1:O", title="New Model"),
        alt.Y("m2:O", title="Baseline Model"),
    ).properties(
        width=500,
        height=400,
        title=title,
    )
    rects = base.mark_rect().encode(
        color="value:Q",
    )
    text = base.mark_text(
        align="center",
        baseline="middle",
        color="black",
        size=12,
        dx=0,
    ).encode(
        text="value:Q",
    )
    return rects + text


def analyse_model():
    """Credit risk analysis: model."""
    preds = analysis_data()
    y_true = preds["y_valid"].values
    y_prob = preds["y_prob"].values

    st.subheader("ANOVA")
    accept_threshold = st.slider(
        "Probability cutoff for approval",
        min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    bad_mean, good_mean, anova = anova_func(y_true, y_prob, accept_threshold)
    st.write(
        f"Mean default rate in group predicted to be bad = `{bad_mean:.4f}`"
    )
    st.write(
        f"Mean default rate in group predicted to be good = `{good_mean:.4f}`"
    )
    st.write(
        "ANOVA statistic for difference in default rate between "
        f"predicted bad and predicted good = `{anova:.4f}`"
    )

    st.subheader("KS Statistic")
    bad_prob, good_prob, ks = ks_func(y_true, y_prob)
    # st.write(f"p-value = `{ks[1]}`")
    st.altair_chart(
        cdf_charts(bad_prob, good_prob).properties(
            title=f"CDFs: KS statistic = {ks[0]:.4f}"
        ),
        use_container_width=True,
    )
    # cdf_charts(bad_prob, good_prob)

    st.subheader("ROC Curve and Gini Coefficient")
    # st.write(f"ROC AUC = `{lrAUC:.4f}`")
    roc_auc, gini, fpr, tpr, _ = roc_func(y_true, y_prob)
    st.altair_chart(
        roc_chart(fpr, tpr).properties(
            title=f"ROC AUC = {roc_auc:.4f}, Gini = {gini:.4f}"
        ),
        use_container_width=True,
    )

    st.subheader("Odds ratio")
    num_bins = 10
    tranche_table = odds_func(y_true, y_prob, num_bins)

    st.bar_chart(
        tranche_table[
            ["Pred Proba of Default", "Odds (good to bad)"]
        ].set_index("Pred Proba of Default")
    )

    # st.subheader("Table of performance per tranche")
    st.table(tranche_table)


def odds_chart(y_true, y_prob, y_baseline, num_bins):
    mod_tranche = odds_func(y_true, y_prob, num_bins)
    df1 = mod_tranche[["Pred Proba of Default", "Odds (good to bad)"]].copy()
    df1["Model"] = "New Model"
    bl_tranche = odds_func(y_true, y_baseline, num_bins)
    df2 = bl_tranche[["Pred Proba of Default", "Odds (good to bad)"]].copy()
    df2["Model"] = "Baseline Model"
    source = pd.concat([df1, df2], axis=0).fillna(0)

    c = alt.Chart(source).mark_bar().encode(
        x=alt.X("Model:N", title=""),
        y="Odds (good to bad):Q",
        color="Model:N",
        column=alt.Column(
            "Pred Proba of Default:N",
            title="Predicted Probability of Default Bin",
        ),
    ).properties(height=200, width=400 / num_bins)
    st.altair_chart(c, use_container_width=False)


def metrics_tables(y_true, y_prob, y_baseline, threshold):
    bad_mean, good_mean, anova = anova_func(y_true, y_prob, threshold)
    _, _, ks = ks_func(y_true, y_prob)
    roc_auc, gini, _, _, _ = roc_func(y_true, y_prob)
    bl_bad_mean, bl_good_mean, bl_anova = anova_func(
        y_true, y_baseline, threshold)
    _, _, bl_ks = ks_func(y_true, y_baseline)
    bl_roc_auc, bl_gini, _, _, _ = roc_func(y_true, y_baseline)
    _df = pd.DataFrame(
        [
            [bad_mean, bl_bad_mean],
            [good_mean, bl_good_mean],
            [anova, bl_anova],
            [ks[0], bl_ks[0]],
            [roc_auc, bl_roc_auc],
            [gini, bl_gini],
        ],
        columns=["New Model", "Baseline Model"],
        index=[
            "Mean default rate in predicted bad group",
            "Mean default rate in predicted good group",
            "ANOVA", "KS", "ROC AUC", "Gini",
        ],
    )
    st.table(_df)


def stats_table(y_true, y_prob, y_baseline, threshold):
    df1 = acct_table(
        y_true, y_prob, threshold, LOSS_PER_BAD_ACCT, REV_PER_GOOD_ACCT)
    df1.columns = ["New Model"]
    df2 = acct_table(
        y_true, y_baseline, threshold, LOSS_PER_BAD_ACCT, REV_PER_GOOD_ACCT)
    df2.columns = ["Baseline Model"]
    output_df = df1.join(df2)
    st.table(output_df)

    x = output_df[output_df.index == "Total Profit"].values[0]
    st.write(f"Net Gain = `${x[0] - x[1]:.2f}`")


def compare_models():
    """Credit risk analysis: model comparison. """
    preds = analysis_data()
    y_true = preds["y_valid"].values
    y_prob = preds["y_prob"].values
    y_baseline = preds["y_baseline"].values

    st.subheader("Score shift")
    num_bins = 10
    score_shift_matrix = score_shift_func(y_prob, y_baseline, num_bins)
    # st.write(score_shift_matrix / len(y_true))
    st.altair_chart(
        heatmap_chart(score_shift_matrix),
        use_container_width=False,
    )

    st.subheader("Odds ratio")
    odds_chart(y_true, y_prob, y_baseline, num_bins)

    accept_threshold = st.slider(
        "Probability cutoff for approval",
        min_value=0.0, max_value=1.0, value=0.05, step=0.01, key=1)

    st.subheader("Metrics")
    metrics_tables(y_true, y_prob, y_baseline, accept_threshold)

    st.subheader("Statistics")
    stats_table(y_true, y_prob, y_baseline, accept_threshold)

    st.subheader("Swap set analysis")
    (
        pct_above, pct_below, pct_swap_above, pct_swap_below,
        odds_swap_above, odds_swap_below
    ) = swapset(
        y_true, y_prob, y_baseline, accept_threshold
    )
    st.table(pd.DataFrame(
        [[pct_below], [pct_above], [pct_swap_below], [pct_swap_above],
         [odds_swap_below], [odds_swap_above]],
        columns=[f"Probability cutoff = {accept_threshold:.2f}"],
        index=[
            "Baseline model: % Below",
            "Baseline model: % Above",
            "Swap set: % Swapped Below",
            "Swap set: % Swapped Above",
            "Swap set odds: Swapped Below",
            "Swap set odds: Swapped Above",
        ]
    ))
    net = pct_swap_below - pct_swap_above
    is_more = "more" if net > 0 else "less"

    st.write(
        f"""
        With probability of default cutoff at `{accept_threshold:.2f}`,
        the baseline model indicate that `{pct_below:.1f}%` of the loans
        had a probability of `{accept_threshold:.2f}` or below, and "
        `{pct_above:.1f}%` had a score below the `{accept_threshold:.2f}`
        cutoff. Analyzing the swap set, we see a net of `{net:.1f}%` "
        of the population that would have moved below a probability of
        `{accept_threshold:.2f}` "
        as a result of switching to new model - i.e., the difference between
        the `{pct_swap_above:.2f}%` of the population that swapped above and
        the `{pct_swap_below:.2f}%` of the population swapped below the
        predicted probability cutoff, resulting in
        {is_more} consumers being approved for credit.
        """
    )
    st.write(
        """
        From the performance odds (good to bad ratio) of the actual loans,
        we see consumers shifting to lower probability ranges that have higher
        odds of repayment and those shifting to higher probability that have
        lower odds of performance, demonstrating that the new model further
        refines classifying risk prediction.
        """
    )


def main():
    st.title("Credit Risk Analysis")

    select = st.sidebar.selectbox(
        label="Select mode", options=["Model Comparison", "New Model"])
    st.header(select)

    if select == "New Model":
        analyse_model()
    elif select == "Model Comparison":
        compare_models()


if __name__ == "__main__":
    main()
