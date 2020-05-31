"""
Script for serving.
"""
import json
import pickle

import numpy as np
from metrics import model_monitor
from flask import Flask,  current_app, request

from preprocess.constants import FEATURES

model = pickle.load(open("/artefact/model.pkl", "rb"))


def predict_score(request_json):
    """Predict function."""
    # Get features
    row_feats = f(request_json)

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
