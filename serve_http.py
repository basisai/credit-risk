"""
Script for serving.
"""
import json
import pickle

import numpy as np
import pandas as pd
from metrics import model_monitor
from flask import Flask,  current_app, request

from preprocess.constants import FEATURES

model = pickle.load(open("/artefact/model.pkl", "rb"))

# Simulate redis store
redis = pd.read_parquet("output/test.gz.parquet")


def read_redis_features(sk_id):
    """Gets all the values from redis."""
    # Simulate querying redis
    row = redis.query(f"SK_ID_CURR == '{sk_id}'")
    if len(row) == 0:
        return None
    return row[FEATURES]


def predict_score(request_json):
    """Predict function."""
    # Get features
    sk_id = request_json["sk_id"]
    row_feats = read_redis_features(sk_id)

    if row_feats is not None:
        # Score
        prob = (
            model
            .predict_proba(np.array(row_feats).reshape(1, -1))[:, 1]
            .item()
        )

        # Log the prediction
        current_app.monitor.log_prediction(
            request_body=json.dumps(request_json),
            features=row_feats,
            output=prob
        )
        return prob

    return np.NaN


# pylint: disable=invalid-name
app = Flask(__name__)
app.register_blueprint(model_monitor)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    return {"prob": predict_score(request.json)}


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
