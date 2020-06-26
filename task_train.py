"""
Script to train model.
"""
import logging
import os
import pickle
import time

from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.analyzer import ModelTypes
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

from preprocess.constants import FEATURES, FEATURES_PRUNED, TARGET, CONFIG_FAI
from preprocess.utils import load_data, get_execution_date

TMP_BUCKET = "gs://span-temp-production/"
# TMP_BUCKET = "data/"

MODEL_VER = os.getenv("MODEL_VER")
NUM_LEAVES = os.getenv("NUM_LEAVES")
MAX_DEPTH = os.getenv("MAX_DEPTH")
OUTPUT_MODEL_PATH = "/artefact/model.pkl"
FEATURE_COLS_PATH = "/artefact/feature_cols.pkl"


def get_feats_to_use():
    if MODEL_VER == "xgboost-pruned" or MODEL_VER == "lightgbm-pruned":
        return FEATURES_PRUNED
    return FEATURES


def get_model():
    if MODEL_VER == "lightgbm" or MODEL_VER == "lightgbm-pruned":
        return lgb.LGBMClassifier(
            num_leaves=int(NUM_LEAVES),
            max_depth=int(MAX_DEPTH),
            learning_rate=0.02,
            n_estimators=10000,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1,
        )
    elif MODEL_VER == "xgboost" or MODEL_VER == "xgboost-pruned":
        print("  NUM_LEAVES not used for xgboost model")
        return xgb.XGBClassifier(
            max_depth=int(MAX_DEPTH),
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
    print("Evaluation\n"
          f"  Accuracy          = {acc:.4f}\n"
          f"  Precision         = {precision:.4f}\n"
          f"  Recall            = {recall:.6f}\n"
          f"  F1 score          = {f1_score:.6f}\n"
          f"  ROC AUC           = {roc_auc:.6f}\n"
          f"  Average precision = {avg_prc:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())

    # Calculate and upload xafai metrics
    analyzer = ModelAnalyzer(clf, 'tree_model', model_type=ModelTypes.TREE).test_features(x_val)
    analyzer.fairness_config(CONFIG_FAI).test_labels(y_val).test_inference(y_pred)
    analyzer.analyze()


def trainer(execution_date):
    """Entry point to perform training."""
    print("Load train data")
    data = load_data(TMP_BUCKET + "credit_train/train.csv")
    data = data.fillna(0)
    print("  Train data shape:", data.shape)

    feature_cols = get_feats_to_use()
    train, valid = train_test_split(data, test_size=0.2, random_state=0)
    x_train = train[feature_cols]
    y_train = train[TARGET].values
    x_valid = valid[feature_cols]
    y_valid = valid[TARGET].values

    print("\nTrain model")
    start = time.time()
    clf = get_model()
    clf.fit(x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_metric='auc',
            verbose=200,
            early_stopping_rounds=200)
    print("  Time taken = {:.0f} s".format(time.time() - start))

    print("\nScore model")
    start = time.time()
    selected = np.random.choice(x_train.shape[0], size=2000, replace=False)
    features = x_train.iloc[selected]
    inference = clf.predict_proba(features)[:, 1]

    print("Log metrics")
    ModelMonitoringService.export_text(
        features=features.iteritems(),
        inference=inference.tolist(),
    )
    print("  Time taken = {:.0f} s".format(time.time() - start))

    print("\nEvaluate")
    compute_log_metrics(clf, x_valid, y_valid)

    print("\nSave model")
    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(clf, model_file)

    # Save feature names
    with open(FEATURE_COLS_PATH, "wb") as file:
        pickle.dump(feature_cols, file)

    # To simulate redis, save to artefact
    from shutil import copyfile
    copyfile("output/test.gz.parquet", "/artefact/test.gz.parquet")


def main():
    execution_date = get_execution_date()
    print(execution_date.strftime("\nExecution date is %Y-%m-%d"))
    trainer(execution_date)


if __name__ == "__main__":
    main()
