"""
Script to perform preprocessing of previous_application data.
"""
import numpy as np
import pandas as pd

from preprocess.utils import load_data, onehot_enc

BUCKET = "gs://bedrock-sample/credit/"
# BUCKET = "data/"

CATEGORICAL_COLS = [
    'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT',
    'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
    'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY',
    'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY',
    'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
]

CATEGORIES = [
    ['Cash loans', 'Consumer loans', 'Revolving loans', 'XNA'],
    ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'],
    ['N', 'Y'],
    [
        'Building a house or an annex', 'Business development', 'Buying a garage',
        'Buying a holiday home / land', 'Buying a home', 'Buying a new car',
        'Buying a used car', 'Car repairs', 'Education', 'Everyday expenses',
        'Furniture', 'Gasification / water supply', 'Hobby', 'Journey', 'Medicine',
        'Money for a third person', 'Other', 'Payments on other loans',
        'Purchase of electronic equipment', 'Refusal to name the goal', 'Repairs',
        'Urgent needs', 'Wedding / gift / holiday', 'XAP', 'XNA',
    ],
    ['Approved', 'Canceled', 'Refused', 'Unused offer'],
    [
        'Cash through the bank', 'Cashless from the account of the employer',
        'Non-cash from your account', 'XNA',
    ],
    ['CLIENT', 'HC', 'LIMIT', 'SCO', 'SCOFR', 'SYSTEM', 'VERIF', 'XAP', 'XNA'],
    ['Children', 'Family', 'Group of people', 'Other_A', 'Other_B', 'Spouse, partner', 'Unaccompanied'],
    ['New', 'Refreshed', 'Repeater', 'XNA'],
    [
        'Additional Service', 'Animals', 'Audio/Video', 'Auto Accessories',
        'Clothing and Accessories', 'Computers', 'Construction Materials',
        'Consumer Electronics', 'Direct Sales', 'Education', 'Fitness', 'Furniture',
        'Gardening', 'Homewares', 'House Construction', 'Insurance', 'Jewelry',
        'Medical Supplies', 'Medicine', 'Mobile', 'Office Appliances', 'Other',
        'Photo / Cinema Equipment', 'Sport and Leisure', 'Tourism', 'Vehicles',
        'Weapon', 'XNA',
    ],
    ['Cards', 'Cars', 'Cash', 'POS', 'XNA'],
    ['XNA', 'walk-in', 'x-sell'],
    [
        'AP+ (Cash loan)', 'Car dealer', 'Channel of corporate sales', 'Contact center',
        'Country-wide', 'Credit and cash offices', 'Regional / Local', 'Stone',
    ],
    [
        'Auto technology', 'Clothing', 'Connectivity', 'Construction', 'Consumer electronics',
        'Furniture', 'Industry', 'Jewelry', 'MLM partners', 'Tourism', 'XNA',
    ],
    ['XNA', 'high', 'low_action', 'low_normal', 'middle'],
    [
        'Card Street', 'Card X-Sell', 'Cash', 'Cash Street: high', 'Cash Street: low',
        'Cash Street: middle', 'Cash X-Sell: high', 'Cash X-Sell: low', 'Cash X-Sell: middle',
        'POS household with interest', 'POS household without interest',
        'POS industry with interest', 'POS industry without interest', 'POS mobile with interest',
        'POS mobile without interest', 'POS other with interest', 'POS others without interest',
    ],
]


def previous_application():
    prev = load_data(BUCKET + 'auxiliary/previous_application.csv')

    # One-hot encoding of categorical features
    prev, cat_cols = onehot_enc(prev, CATEGORICAL_COLS, CATEGORIES)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    return prev_agg
