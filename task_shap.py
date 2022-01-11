"""
Script to perform batch scoring of Shapley values.
"""
import pickle
from datetime import datetime

import bdrk
import pandas as pd
import shap

from preprocess.utils import get_execution_date, get_temp_bucket_prefix
from preprocess.constants import TARGET

TMP_BUCKET = get_temp_bucket_prefix()


def load_data(execution_date: datetime) -> pd.DataFrame:
    """Load data."""
    # To simulate loading data by execution_date from saved data
    # partitioned by date, load previously saved test data
    return pd.read_parquet("/artefact/test.gz.parquet")


def compute_shap(execution_date: datetime) -> None:
    """Batch scoring pipeline"""
    print("\nLoad data")
    data = load_data(execution_date)

    print("\nLoad model")
    model = pickle.load(open("/artefact/model.pkl", "rb"))
    feature_cols = pickle.load(open("/artefact/feature_cols.pkl", "rb"))

    print("\nCompute Shapley values")
    explainer = shap.TreeExplainer(model)
    # Taking the shapley values that correspond to TARGET=1
    shap_values = explainer.shap_values(data[feature_cols])[1]
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["SK_ID_CURR"] = data["SK_ID_CURR"].values

    print("\nCompute predicted probability")
    output_df = data[["SK_ID_CURR", TARGET]].copy()
    # Taking the predicted probability that corresponds to TARGET=1
    output_df["Probability"] = model.predict_proba(data[feature_cols])[:, 1]
    output_df = pd.merge(output_df, shap_df, on="SK_ID_CURR")
    print("  Output data shape:", output_df.shape)

    print("\nSave output data")
    output_df.to_csv(f"{TMP_BUCKET}/credit_shap/shap.csv", index=False)


def main() -> None:
    execution_date = get_execution_date()
    print(execution_date.strftime("\nExecution date is %Y-%m-%d"))
    compute_shap(execution_date)


if __name__ == "__main__":
    bdrk.init()
    with bdrk.start_run():
        main()
