"""
Script to train model.
"""
import logging
import os
import pickle
import time
from datetime import timedelta

import pandas as pd
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

from preprocess.constants import FEATURES, TARGET
from preprocess.utils import get_execution_date

BUCKET = "gs://span-temp-production/"
# BUCKET = "data/"

MODEL_VER = os.getenv("MODEL_VER")
LR = float(os.getenv("LR"))
OUTPUT_MODEL_PATH = "/artefact/model.pkl"


def get_model():
    if MODEL_VER == "lightgbm":
        clf = lgb.LGBMClassifier(
            nthread=-1,
            num_leaves=34,
            learning_rate=LR,
            n_estimators=10000,
        )
    
    clf = xgb.XGBClassifier(
        nthread=-1,
        num_leaves=34,
        learning_rate=LR,
        n_estimators=10000,
    )
    return clf
    

def compute_log_metrics(clf, x_val, y_val):
    """Compute and log metrics."""
    y_prob = clf.predict_proba(x_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)
    print("Evaluation"
          f"  Accuracy          = {acc:.4f}"
          f"  Precision         = {precision:.4f}"
          f"  Recall            = {recall:.6f}"
          f"  F1 score          = {f1_score:.6f}"
          f"  ROC AUC           = {roc_auc:.6f}"
          f"  Average precision = {avg_prc:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    return {"Model version": MODEL_VER, "Accuracy": acc, "ROC AUC": roc_auc, "Avg precision": avg_prc}


def trainer(execution_date):
    """Entry point to perform training."""
    print("Load train data")
    train_date = (execution_date - timedelta(days=1)).strftime("%Y-%m-%d")
    train_dir = BUCKET + "train_data/date_partition={}/".format(train_date)
    data = pd.read_parquet(train_dir + "train.gz.parquet")
    print("  Train_data shape =", data.shape)
    
    train, valid = train_test_split(data, test_size=0.2, random_state=0)
    x_train, y_train = train[FEATURES], train[TARGET]
    x_valid, y_valid = valid[FEATURES], valid[TARGET]

    print("Train model")
    start = time.time()
    clf = get_model()
    clf.fit(x_train, y_train)
    print("  Time taken = {:.0f} s".format(time.time() - start))

    print("Score model")
    start = time.time()
    selected = np.random.choice(x_train.shape[0], size=20000, replace=False)
    features = x_train.iloc[selected]
    inference = clf.predict_proba(features)[:, 1]

    print("Log metrics")
    ModelMonitoringService.export_text(
        features=features.iteritems(),
        inference=inference.tolist(),
    )
    print("  Time taken = {:.0f} s".format(time.time() - start))

    print("Save model")
    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(clf, model_file)

    print("Evaluate")
    compute_log_metrics(clf, x_valid, y_valid)


def main():
    execution_date = get_execution_date()
    print(execution_date.strftime("\nExecution date is %Y-%m-%d"))
    trainer(execution_date)


if __name__ == "__main__":
    main()
