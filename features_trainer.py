"""
Script to generate features for training.
"""
import os
from datetime import timedelta

from preprocess.application import application
from preprocess.bureau_and_balance import bureau_and_balance
from preprocess.previous_application import previous_application
from preprocess.pos_cash import pos_cash
from preprocess.installments_payments import installments_payments
from preprocess.credit_card_balance import credit_card_balance
from preprocess.utils import timer, get_execution_date

BUCKET = "gs://span-temp-production/"
# BUCKET = "data/"


def generate_features(execution_date):
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

    print("\nSave train data")
    print("  Train data shape:", df.shape)
    train_date = (execution_date - timedelta(days=1)).strftime("%Y-%m-%d")
    train_dir = BUCKET + "train_data/date_partition={}/".format(train_date)
    os.mkdir(train_dir)
    df.to_parquet(
        train_dir + "train.gz.parquet",
        engine="fastparquet",
        compression="gzip",
    )

    
def main():
    execution_date = get_execution_date()
    print(execution_date.strftime("\nExecution date is %Y-%m-%d"))
    generate_features(execution_date)
    

if __name__ == "__main__":
    main()
