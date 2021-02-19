"""
Script to perform preprocessing of bureau and bureau_balance data.
"""
import pandas as pd

from preprocess.utils import load_data, onehot_enc

BUCKET = "s3://bedrock-sample/credit/"
# BUCKET = "data/"

BUREAU_CATEGORICAL_COLS = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

BUREAU_CATEGORIES = [
    ['Active', 'Bad debt', 'Closed', 'Sold'],
    ['currency 1', 'currency 2', 'currency 3', 'currency 4'],
    [
        'Another type of loan',
        'Car loan',
        'Cash loan (non-earmarked)',
        'Consumer credit',
        'Credit card',
        'Interbank credit',
        'Loan for business development',
        'Loan for purchase of shares (margin lending)',
        'Loan for the purchase of equipment',
        'Loan for working capital replenishment',
        'Microloan',
        'Mobile operator loan',
        'Mortgage',
        'Real estate loan',
        'Unknown type of loan',
    ],
]

BB_CATEGORICAL_COLS = ['STATUS']

BB_CATEGORIES = [
    ['0', '1', '2', '3', '4', '5', 'C', 'X'],
]


def bureau_and_balance():
    bureau = load_data(BUCKET + 'auxiliary/bureau.csv')
    bb = load_data(BUCKET + 'auxiliary/bureau_balance.csv')

    # One-hot encoding of categorical features
    bureau, bureau_cat = onehot_enc(bureau, BUREAU_CATEGORICAL_COLS, BUREAU_CATEGORIES)
    bb, bb_cat = onehot_enc(bb, BB_CATEGORICAL_COLS, BB_CATEGORIES)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    return bureau_agg
