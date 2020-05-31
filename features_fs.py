"""
Script to generate features for serving.
"""
from preprocess.utils import get_execution_date


def features_fs(execution_date):
    """Entry point to preprocess raw data."""

    # This is only a simulation step of generating features
    # to be saved in a Redis store for use in serving later.
    import time
    time.sleep(60)


def main():
    execution_date = get_execution_date()
    print(execution_date.strftime("\nExecution date is %Y-%m-%d"))
    features_fs(execution_date)


if __name__ == "__main__":
    main()
