import os
import time
from datetime import datetime, timedelta
from dateutil import parser

import pytz
import pandas as pd
from contextlib import contextmanager
from sklearn.preprocessing import OneHotEncoder


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("  Time taken for {} = {:.0f}s".format(title, time.time() - t0))
    

def get_execution_date():
    """Get execution date using server time."""
    execution_date = os.getenv("EXECUTION_DATE")

    # Run DAG daily at 01H00 UTC = 09H00 SGT
    if not execution_date:
        dt_now = datetime.now(pytz.utc) - timedelta(days=1)
        # If time now is between 00H00 UTC and 01H00 UTC, set execution date 1 more day earlier
        if dt_now.strftime("%H:%M") < "01:00":
            dt_now = dt_now - timedelta(days=1)
        execution_date = parser.parse(dt_now.strftime("%Y-%m-%d"))
    else:
        execution_date = parser.parse(execution_date[:10])

    return execution_date


def onehot_enc(df, categorical_columns, categories):
    """One-hot encoding of categorical columns."""
    noncategorical_cols = [col for col in df.columns if col not in categorical_columns]
    
    enc = OneHotEncoder(categories=categories,
                        sparse=False,
                        handle_unknown='ignore')
    y = enc.fit_transform(df[categorical_columns])
    
    ohe_cols = [
        f"{col}_{c}" for col, cats in zip(categorical_columns, categories) for c in cats]
    df1 = pd.DataFrame(y, columns=ohe_cols)
    
    output_df = pd.concat([df[noncategorical_cols], df1], axis=1)
    return output_df, ohe_cols
