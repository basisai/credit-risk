import json

# List numeric and categorical features
NUMERIC_FEATS = json.load(open("preprocess/numeric_feats.txt"))

# For each categorical feature, get the one-hot encoded feature names
CATEGORY_MAP = json.load(open("preprocess/category_map.txt"))

CATEGORICAL_FEATS = list(CATEGORY_MAP.keys())

OHE_CAT_FEATS = []
for f in CATEGORICAL_FEATS:
    OHE_CAT_FEATS.extend(CATEGORY_MAP[f])

# Train & validation features and target
FEATURES = OHE_CAT_FEATS + NUMERIC_FEATS

# [LightGBM] [Fatal] Do not support special JSON characters in feature name.
FEATURES = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in FEATURES]

TARGET = 'TARGET'

# Pruned features
FEATURES_PRUNED = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'APPROVED_CNT_PAYMENT_MEAN',
    'AMT_ANNUITY', 'PAYMENT_RATE', 'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'DAYS_BIRTH',
    'NAME_FAMILY_STATUS_Married', 'AMT_GOODS_PRICE', 'NAME_EDUCATION_TYPE_Higher_education',
    'INSTAL_DPD_MEAN', 'PREV_CNT_PAYMENT_MEAN', 'INSTAL_AMT_PAYMENT_SUM', 'ANNUITY_INCOME_PERC',
    'AMT_CREDIT', 'PREV_APP_CREDIT_PERC_MEAN', 'FLAG_PHONE', 'CLOSED_DAYS_CREDIT_ENDDATE_MAX',
    'FLAG_DOCUMENT_3', 'INSTAL_DAYS_ENTRY_PAYMENT_SUM', 'APPROVED_AMT_DOWN_PAYMENT_MAX',
    'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'INSTAL_AMT_PAYMENT_MIN',
    'INSTAL_AMT_INSTALMENT_SUM', 'ACTIVE_DAYS_CREDIT_MAX', 'DAYS_EMPLOYED_PERC',
    'POS_MONTHS_BALANCE_SIZE', 'BURO_CREDIT_ACTIVE_Closed_MEAN',
    'PREV_NAME_YIELD_GROUP_low_action_MEAN', 'INSTAL_PAYMENT_PERC_MEAN',
    'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 'DAYS_ID_PUBLISH', 'NAME_INCOME_TYPE_Working',
    'BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'PREV_NAME_YIELD_GROUP_high_MEAN',
    'DEF_30_CNT_SOCIAL_CIRCLE', 'PREV_NAME_CLIENT_TYPE_New_MEAN', 'AMT_REQ_CREDIT_BUREAU_QRT',
    'PREV_NAME_CONTRACT_STATUS_Refused_MEAN', 'INSTAL_PAYMENT_DIFF_MEAN',
    'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'INCOME_CREDIT_PERC', 'INSTAL_DBD_SUM',
    'ACTIVE_AMT_CREDIT_SUM_SUM', 'PREV_APP_CREDIT_PERC_MIN', 'REFUSED_DAYS_DECISION_MAX',
    'INSTAL_PAYMENT_DIFF_MAX',
]


CONFIG_FAI = {
    'CODE_GENDER': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [1],
    },
    'NAME_EDUCATION_TYPE_Higher education': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [1],
    },
}
