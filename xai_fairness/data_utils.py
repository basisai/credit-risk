"""
Utility functions for streamlit app.
"""
from distutils.util import strtobool

import numpy as np
import pandas as pd
import streamlit as st

from xai_fairness.toolkit_xai import get_explainer, compute_shap


class DataInputs:
    def __init__(
            self,
            config,
            feature_names,
            target_name,
            target_classes=None,
            protected_features=None,
        ):
        """
        Data for the dashboard.

        Args:
            config: configuration
            feature_names list[str]: list of feature names
            target_name str
            target_classes list[str]: list of target labels
            protected_features Dict: protected features for fairness
        """
        self.config = config
        self.protected_features = protected_features
        self.feature_names = feature_names
        self.target_name = target_name
        self.target_classes = target_classes
        self.ml_type = config["ml_type"]
        if isinstance(config["is_multiclass"], bool):
            self.is_multiclass = config["is_multiclass"]
        else:
            self.is_multiclass = bool(strtobool(config["is_multiclass"]))

        self.model = None

        self.valid_fai = None
        self.true_class = None
        self.pred_class = None
        self.max_rows = 3000
        self.has_fai = bool(protected_features.keys()) if protected_features is not None else False

    def model(self, model):
        self.model = model
        return self

    def xai_data(self, shap_summary_dfs, shap_sample_dfs, base_values, pred_sample_df):
        """
        Sample validation data for generating XAI plots only.
        Data will be limited by sampling 3000 rows randomly.

        Args:
            valid_features pd.dataFrame: sample validation data
            shap_summary_dfs list[pd.dataFrame]: list of dataframes containing
                average absolute SHAP values and correlations
            shap_sample_dfs
            base_values numpy.array
            pred_sample_df pd.dataFrame
        """
        self.shap_summary_dfs = shap_summary_dfs
        self.shap_sample_dfs = shap_sample_dfs
        self.base_values = base_values
        self.pred_sample_df = pred_sample_df
        return self

    def fai_data(self, valid_fai=None, true_class=None, pred_class=None):
        """
        To be used for fairness
        The classes need not correspond to the original target variable.
        Eg, for regression, one can bin the scores to classes for computing fairness metrics.

        Args:
            valid_fai pd.DataFrame: validation data containing protected feature columns
            indicated in protected_features
            true_class numpy.array: actual labels
            pred_class numpy.array: labels from predicted scores
        """
        self.valid_fai = valid_fai
        self.true_class = true_class
        self.pred_class = pred_class
        return self

    def check_data(self):
        if self.valid_fai is None and self.has_fai:
            raise AttributeError("Data for fairness is missing")

        if self.ml_type not in ["regression", "classification"]:
            raise ValueError("ml_type has to be either 'regression' or 'classification'")

        if self.ml_type == "regression" and self.is_multiclass:
            raise ValueError("ml_type is set as 'regression' and is_multiclass is set as True")

        if self.is_multiclass and len(self.target_classes) < 3:
            raise ValueError("is_multiclass is set as True but there are less than 3 target classes")


def sampling(df, max_rows=3000, random_state=0):
    """Select first 'max_rows' rows for plotting purposes."""
    if df is not None and df.shape[0] > max_rows:
        return df.sample(max_rows, random_state=random_state)
    return df
