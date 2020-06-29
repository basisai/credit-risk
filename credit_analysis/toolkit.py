"""
Toolkit
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import metrics
from scipy.sparse import coo_matrix


def anova_func(y_true, y_prob, threshold):
    """Compute ANOVA."""
    anova_cut = y_prob > threshold
    predicted_bad = y_true[anova_cut]
    predicted_good = y_true[~anova_cut]
    bad_mean = np.mean(predicted_bad)
    good_mean = np.mean(predicted_good)
    anova_stat = (bad_mean - good_mean) ** 2 / (
            0.5 * (np.var(predicted_good) + np.var(predicted_bad)))
    return bad_mean, good_mean, anova_stat


def ks_func(y_true, y_prob):
    """Compute KS statistics."""
    bad_prob = y_prob[y_true == 1]
    bad_prob = np.sort(bad_prob)
    good_prob = y_prob[y_true == 0]
    good_prob = np.sort(good_prob)
    ks_stat = stats.ks_2samp(bad_prob, good_prob)
    return bad_prob, good_prob, ks_stat


def roc_func(y_true, y_prob):
    """Compute ROC AUC and Gini."""
    roc_auc = metrics.roc_auc_score(y_true, y_prob)
    gini = roc_auc * 2 - 1
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    return roc_auc, gini, fpr, tpr, thresholds


def odds_func(y_true, y_prob, num_bins):
    df = pd.DataFrame({
        "actual": y_true,
        "pred": y_prob,
    })
    df["bin1"] = np.clip((y_prob * num_bins).astype(int), 0, num_bins - 1)
    df1 = df.groupby("bin1").agg({"actual": ["count", "sum"]})
    df1.columns = ['Number of loans', 'Number of bad loans']
    df1 = pd.DataFrame({'Pred Proba of Default': [
        f"{i / num_bins:.1f} to {(i + 1) / num_bins:.1f}" for i in range(num_bins)]}).join(df1)
    df1.fillna(0, inplace=True)
    df1 = df1[::-1].reset_index(drop=True)
    df1["Cumulative % of loans"] = np.cumsum(df1["Number of loans"]) / len(y_true) * 100
    df1["Cumulative % of bad loans"] = np.cumsum(df1["Number of bad loans"]) / \
                                       df1["Number of bad loans"].sum() * 100
    df1["Odds (good to bad)"] = (df1['Number of loans'] - df1['Number of bad loans']) / \
                                df1['Number of bad loans']
    return df1[['Pred Proba of Default', 'Number of loans', 'Cumulative % of loans',
                'Number of bad loans', 'Cumulative % of bad loans', 'Odds (good to bad)']]


def score_shift_func(prob_a, prob_b, num_bins):
    bin1 = np.clip((prob_a * num_bins).astype(int), 0, num_bins - 1)
    bin2 = np.clip((prob_b * num_bins).astype(int), 0, num_bins - 1)
    score_mat = coo_matrix(
        (np.ones(len(bin1)), (bin1, bin2)),
        shape=(num_bins, num_bins)
    ).toarray()
    axnames = [f"{i / num_bins:.1f} to {(i + 1) / num_bins:.1f}" for i in range(num_bins)]
    score_shift_matrix = pd.DataFrame(score_mat, columns=axnames, index=axnames)
    return score_shift_matrix


def acct_table(y_true, y_prob, threshold, loss_per_bad_acct=2500, rev_per_good_acct=100):
    accepts = y_prob < threshold
    num_bad_accts = np.sum(y_true[accepts])
    num_good_accts = np.sum(accepts) - num_bad_accts
    rev = num_good_accts * rev_per_good_acct
    loss = num_bad_accts * loss_per_bad_acct

    output_df = pd.DataFrame([
        ['Number of Applicants', len(y_true)],
        ['Number of Accepts', np.sum(accepts)],
        ['Acceptance rate', np.mean(accepts)],
        ['Number of Bad accounts', num_bad_accts],
        ['Bad rate', num_bad_accts / np.sum(accepts)],
        ['Estimated Loss Per Bad Account', loss_per_bad_acct],
        ['Estimated Revenue Per Good Account', rev_per_good_acct],
        ['Number of Good accounts', num_good_accts],
        ['Total Revenue', rev],
        ['Total Loss', loss],
        ['Total Profit', rev - loss],
    ], columns=['Metric', 'Model'])
    output_df.set_index("Metric", inplace=True)
    return output_df


def swapset(y_true, y_prob, y_baseline, cutoff):
    cnt = np.sum(y_baseline > cutoff)
    swap_above = np.sum((y_baseline < cutoff) & (y_prob > cutoff))
    swap_above_bad = np.sum((y_baseline < cutoff) & (y_prob > cutoff) & (y_true == 1))
    swap_below = np.sum((y_baseline > cutoff) & (y_prob < cutoff))
    swap_below_bad = np.sum((y_baseline > cutoff) & (y_prob < cutoff) & (y_true == 1))

    pct_above = cnt / len(y_true) * 100
    pct_below = 100 - cnt / len(y_true) * 100
    pct_swap_above = swap_above / len(y_true) * 100
    pct_swap_below = swap_below / len(y_true) * 100
    odds_swap_above = (swap_above - swap_above_bad) / swap_above_bad
    odds_swap_below = (swap_below - swap_below_bad) / swap_below_bad
    return pct_above, pct_below, pct_swap_above, pct_swap_below, odds_swap_above, odds_swap_below


# TODO
# def swapset_tables(y_true, y_prob, y_baseline, swapSetThreshold):
#     lrAccepts = (y_prob < swapSetThreshold)
#     tmAccepts = (y_baseline < swapSetThreshold)
#
#     st.write('Original Breakdown:')
#     from sklearn import metrics
#     accept_mat = metrics.confusion_matrix(lrAccepts * 1, tmAccepts * 1)
#     preSwapTable = pd.DataFrame(np.rot90(np.rot90(accept_mat)),
#                                 columns=['Accepted by B', 'Rejected by B'],
#                                 index=['Accepted by A', 'Rejected by A'])
#     st.write(preSwapTable)
#
#     # preSwapTable = np.zeros((2, 2))
#     # preSwapTable[0, 0] = np.sum(lrAccepts[lrAccepts == tmAccepts])
#     # preSwapTable[1, 1] = np.sum(~lrAccepts[lrAccepts == tmAccepts])
#     # preSwapTable[0, 1] = np.sum(lrAccepts[lrAccepts != tmAccepts])
#     # preSwapTable[1, 0] = np.sum(tmAccepts[lrAccepts != tmAccepts])
#     # preSwapTable = pd.DataFrame(preSwapTable, columns=['Accepted by B', 'Rejected by B'],
#     #                             index=['Accepted by A', 'Rejected by A'])
#     # st.write(preSwapTable)
#
#     st.write('Swapped to maintain same approval rate:')
#     swapSameApprovalRate = np.zeros((3, 3))
#     swapSameApprovalRate[0, 0] = np.sum(lrAccepts[lrAccepts == tmAccepts])
#     swapSameApprovalRate[0, 1] = np.sum(lrAccepts[lrAccepts != tmAccepts])
#     swapSameApprovalRate[1, 0] = np.sum(lrAccepts[lrAccepts != tmAccepts])
#     swapSameApprovalRate[1, 1] = np.sum(~lrAccepts[lrAccepts == tmAccepts])
#     swapSameApprovalRate[0, 2] = np.sum(y_true[lrAccepts]) / np.sum(lrAccepts)
#
#     lr_num = lrAccepts[lrAccepts != tmAccepts].sum()
#     tm_num = tmAccepts[lrAccepts != tmAccepts].sum()
#     if lr_num < tm_num:
#         swapIn = np.random.choice(
#             y_true[(lrAccepts != tmAccepts) & (tmAccepts == 1)], lr_num, replace=False)
#         swapSameApprovalRate[2, 0] = (np.sum(y_true[(lrAccepts == tmAccepts) & (lrAccepts == 1)]) +
#                                       np.sum(swapIn)) / \
#                                      (swapSameApprovalRate[1, 0] + swapSameApprovalRate[0, 0])
#     swapSameApprovalRate = pd.DataFrame(swapSameApprovalRate,
#                                         columns=['Accepted by B', 'Rejected by B', 'Old Bad Rate'],
#                                         index=['Accepted by A', 'Rejected by A', 'New Bad Rate'])
#
#     swapSameApprovalRate = swapSameApprovalRate.style.apply(highlight_cells, axis=None)
#     st.write(swapSameApprovalRate)
#     st.write('*Legend: Green - new bad rate, Red - old bad rate*')


def highlight_cells(x):
    df = x.copy()
    # set default color
    # df.loc[:,:] = 'background-color: papayawhip'
    df.loc[:, :] = ''
    # set particular cell colors
    df.iloc[0, 2] = 'background-color: red'
    df.iloc[2, 0] = 'background-color: lightgreen'
    return df
