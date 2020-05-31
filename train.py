"""
Script to train model.
"""
import logging
import os
import pickle
import time

from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

from preprocess.constants import FEATURES, TARGET
from preprocess.utils import load_data, get_execution_date

TMP_BUCKET = "gs://span-temp-production/"
# TMP_BUCKET = "data/"

MODEL_VER = os.getenv("MODEL_VER")
NUM_LEAVES = int(os.getenv("NUM_LEAVES"))
OUTPUT_MODEL_PATH = "/artefact/model.pkl"


def get_model():
    if MODEL_VER == "lightgbm":
        return lgb.LGBMClassifier(
            num_leaves=NUM_LEAVES,
            learning_rate=0.02,
            n_estimators=10000,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1,
        )
    elif MODEL_VER == "xgboost":
        return xgb.XGBClassifier(
            num_leaves=NUM_LEAVES,
            learning_rate=0.02,
            n_estimators=10000,
        )
    else:
        raise Exception("Model not implemented")
    

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
    data = load_data(TMP_BUCKET + "credit_train/train.csv")
    print("  Train data shape:", data.shape)
    
    train, valid = train_test_split(data, test_size=0.2, random_state=0)
    x_train = train[FEATURES]
    y_train = train[TARGET].values
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    # # [LightGBM] [Fatal] Do not support special JSON characters in feature name.
    # new_cols = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in x_train.columns]
    # x_train.columns = new_cols
    # x_valid.columns = new_cols

    print("Train model")
    start = time.time()
    clf = get_model()
    clf.fit(x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_metric='auc',
            verbose=200,
            early_stopping_rounds=200)
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
