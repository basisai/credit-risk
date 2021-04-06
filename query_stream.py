import time
import json
import requests
import random

import numpy as np
import pandas as pd

URL = "<MODEL_ENDPOINT_URL>"
TOKEN = "<MODEL_ENDPOINT_TOKEN>"


def get_data(sk_ids):
    data = {"sk_id": random.choice(sk_ids)}
    return json.dumps(data)


if __name__ == "__main__":
    test = pd.read_parquet("./data/test.gz.parquet")
    sk_ids = test["SK_ID_CURR"].tolist()

    headers = {
        'Content-Type': 'application/json',
        'X-Bedrock-Api-Token': TOKEN,
    }

    print("Start query stream")
    num_queries = 60
    start = time.time()
    for i in range(1, num_queries + 1):
        if i % 10 == 0:
            ttaken = time.time() - start
            print(f"Time taken = {ttaken:.0f}s, Rate = {i / ttaken:.3f} queries/s")
        requests.post(URL, headers=headers, data=get_data(sk_ids))
        time.sleep(np.random.randint(50, 70) / 60 * 2.5)

    ttaken = time.time() - start
    print(f"Total time taken = {ttaken:.0f}s, Rate = {num_queries / ttaken:.3f} queries/s")
