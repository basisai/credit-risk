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
from bedrock_client import bdrk

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
          f"  Recall            = {recall:.4f}\n"
          f"  F1 score          = {f1_score:.4f}\n"
          f"  ROC AUC           = {roc_auc:.4f}\n"
          f"  Average precision = {avg_prc:.4f}")

    # Remove BedrockApi client instantiation
    #   - Replaced by `bdrk.init` (see `bedrock_main` below).
    #   - Simplifies user code to not have to pass the API client object
    #     around.
    #   - Layer of indirection to facilitate migrating to new backend routes
    #     without disrupting existing orchestration users (who may continue to
    #     use `BedrockApi`).
    #   - Con: Cannot concurrently log to different backend servers, projects,
    #     or runs in the same program.

    # Log metrics
    bdrk.log_metric("Accuracy", acc)
    bdrk.log_metric("ROC AUC", roc_auc)
    bdrk.log_metric("Avg precision", avg_prc)
    bdrk.log_chart_data(y_val.astype(int).tolist(),
                        y_prob.flatten().tolist())

    # Calculate and upload xafai metrics
    analyzer = ModelAnalyzer(clf, 'tree_model', model_type=ModelTypes.TREE).test_features(x_val)
    analyzer.fairness_config(CONFIG_FAI).test_labels(y_val).test_inference(y_pred)
    analyzer.analyze()


def trainer(execution_date):
    """Entry point to perform training."""
    print("\nLoad train data")
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

    print("\nEvaluate")
    compute_log_metrics(clf, x_valid, y_valid)

    print("\nLog model monitoring metrics")
    start = time.time()
    selected = np.random.choice(x_train.shape[0], size=2000, replace=False)
    features = x_train.iloc[selected]
    inference = clf.predict_proba(features)[:, 1]

    ModelMonitoringService.export_text(
        features=features.iteritems(),
        inference=inference.tolist(),
    )
    print("  Time taken = {:.0f} s".format(time.time() - start))

    print("\nSave model")
    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(clf, model_file)

    # Save feature names
    with open(FEATURE_COLS_PATH, "wb") as file:
        pickle.dump(feature_cols, file)

    # To simulate redis, save to artefact
    from shutil import copyfile
    copyfile("output/test.gz.parquet", "/artefact/test.gz.parquet")

    # log_model: Uploads a model artefact that has been saved to a local path.
    #   - `path` is the local path that will be zipped and uploaded.
    #   - Model is uploaded for the step and should only be called once per
    #     step. Repeated calls result in an error.
    #   - For orchestrated runs:
    #       - Phase 1: Optional. No-op if present.
    #       - Phase 2: Optional. If present, does the upload and suppresses
    #         automatic upload from /artefact (by the Geophone sidecar).
    #       - Phase 3: Required.
    #       - Note: Step artefacts will continue to be aggregated in the final
    #         step of orchestrated runs. Multistep is not currently supported
    #         when using client library.
    bdrk.log_model(path="/artefact")

    # download_model: Downloads artefacts for a Model Version.
    #   - `path` is the destination path where artefacts are downloaded to. If
    #     unspecified, will use the current working directory.
    #   - `log_dependency` specifies whether to log the downloaded Model
    #     Version as an upstream dependency of the current run. A run can have
    #     multiple upstream models.
    #   - Note that `model` refers to Model Collection name.
    bdrk.download_model(
        model="my-model-name",
        version=7,
        path="/model",
        log_dependency=True)


def main():
    execution_date = get_execution_date()
    print(execution_date.strftime("\nExecution date is %Y-%m-%d"))
    trainer(execution_date)

def bedrock_main():
    # init: Initialise bedrock library and get/create project.
    #   - If project does not exist, it will be created.
    #   - Organisation is derived from user, which is from API token.
    #   - Repeated calls will update the existing configuration.
    bdrk.init(
        access_token="...",
        environment="staging",   # Tells Bedrock where to upload artefacts to.
        project="my-project",
        logger=None)

    # `init` is optional for orchestrated runs:
    #   - Automatically called with no params when any `bdrk` function is used.
    #   - Will use env vars for params that are unspecified: server_uri,
    #     api_token, environment, project.
    #   - If explicitly called, mismatch between parameters and env vars will
    #     fail the run. (This avoids any confusion when reviewing the executed
    #     code.)
    #   - In the future, we can consider making this required to be more
    #     consistent across runtimes.

    # start_run: Dynamically get/create training pipeline + model, and start a
    #            new run with a single step.
    #   - If pipeline already exists, a new run will be created under it.
    #   - If pipeline does not exist, a new pipeline and model collection will
    #     be created first.
    #   - For existing pipeline ID, mismatch between model ID and `model`
    #     param will result in run failure.
    with bdrk.start_run(
        pipeline="my-pipeline",  # Pipeline ID
        model=None,              # Leave blank to use pipeline ID
    ):
        # Run and step status updates:
        #   - RUNNING, when entering this block.
        #   - STOPPED, when exiting from this block with a KeyboardInterrupt.
        #   - FAILED, when exiting from this block with another exception.
        #   - SUCCEEDED, when exiting from this block without exceptions.
        #   - Bedrock will create a new Model Version on run SUCCEEDED.
        #   - If program terminates without exiting the block, e.g. OOM killed,
        #     then the user has to update the run status via UI. (We can look
        #     into implementing signal handlers in the future.)

        # `start_run` is optional for orchestrated runs:
        #   - No-op, since pipeline/model/run creation and status updates are
        #     handled by Bedrock.
        #   - Will use env vars for params that are unspecified: pipeline.
        #   - If explicitly called, mismatch between parameters and env vars
        #     will fail the run. (This avoids any confusion when reviewing the
        #     executed code.)
        #   - In the future, we can consider making this required to be more
        #     consistent across runtimes.

        # log_param: Log a key-value pair representing a script parameter.
        #   - Overwrites existing keys.
        #   - For orchestrated runs, parameters passed via env var will
        #     automatically be logged.
        bdrk.log_param("MODEL_VER", MODEL_VER)
        bdrk.log_param("NUM_LEAVES", NUM_LEAVES)
        bdrk.log_param("MAX_DEPTH", MAX_DEPTH)

        main()

if __name__ == "__main__":
    bedrock_main()
