import json

# For explainability AI app
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

TARGET = 'TARGET'


# For fairness AI app
# List bias and privileged info
CONFIG_FAI = {
    'CODE_GENDER': {
        'unprivileged_attribute_values': [1],
        'privileged_attribute_values': [0],
    },
    'NAME_EDUCATION_TYPE_Higher education': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [1],
    },
}
