"""
Script to perform preprocessing of credit_card_balance data.
"""
import pandas as pd

from preprocess.utils import load_data, onehot_enc, get_bucket_prefix

BUCKET = f"{get_bucket_prefix()}/credit"

CATEGORICAL_COLS = ['NAME_CONTRACT_STATUS']

CATEGORIES = [
    [
        'Active', 'Approved', 'Completed', 'Demand',
        'Refused', 'Sent proposal', 'Signed'
    ],
]


def credit_card_balance() -> pd.DataFrame:
    """Preprocess credit card balance."""
    cc = load_data(f'{BUCKET}/auxiliary/credit_card_balance.csv')

    # One-hot encoding of categorical features
    cc, _ = onehot_enc(cc, CATEGORICAL_COLS, CATEGORIES)

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index([
        'CC_' + e[0] + "_" + e[1].upper()
        for e in cc_agg.columns.tolist()
    ])

    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    return cc_agg
