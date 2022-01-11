"""
Script to generate features for training.
"""
from datetime import datetime

import bdrk
from preprocess.application import application
from preprocess.bureau_and_balance import bureau_and_balance
from preprocess.previous_application import previous_application
from preprocess.pos_cash import pos_cash
from preprocess.installments_payments import installments_payments
from preprocess.credit_card_balance import credit_card_balance
from preprocess.utils import timer, get_execution_date, get_temp_bucket_prefix

TMP_BUCKET = get_temp_bucket_prefix()


def generate_features(execution_date: datetime) -> None:
    """Generate features."""
    print("\nProcess application")
    df = application(execution_date)
    print("  Application df shape:", df.shape)

    print("\nProcess bureau and bureau_balance")
    with timer("processing bureau and bureau_balance"):
        bureau = bureau_and_balance()
        print("  Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')

    print("\nProcess previous application")
    with timer("processing previous application"):
        prev = previous_application()
        print("  Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')

    print("\nProcess POS-CASH balance")
    with timer("processing POS-CASH balance"):
        pos = pos_cash()
        print("  Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')

    print("\nProcess installments payments")
    with timer("processing installments payments"):
        ins = installments_payments()
        print("  Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')

    print("\nProcess credit card balance")
    with timer("processing credit card balance"):
        cc = credit_card_balance()
        print("  Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')

    # [LightGBM] [Fatal] Do not support special JSON characters
    # in feature name.
    new_cols = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns
    ]
    df.columns = new_cols

    print("\nSave train data")
    print("  Train data shape:", df.shape)
    df.to_csv(f"{TMP_BUCKET}/credit_train/train.csv", index=False)


def main() -> None:
    execution_date = get_execution_date()
    print(execution_date.strftime("\nExecution date is %Y-%m-%d"))
    generate_features(execution_date)


if __name__ == "__main__":
    bdrk.init()
    with bdrk.start_run():
        main()
