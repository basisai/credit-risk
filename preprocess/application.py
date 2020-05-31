"""
Script to perform preprocessing of application data.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from .constants import TARGET
from .utils import onehot_enc

BUCKET = "gs://bedrock-sample/credit/"
# BUCKET = "data/"

BINARY_MAP = {
    'CODE_GENDER': ['M', 'F'],
    'FLAG_OWN_CAR': ['N', 'Y'],
    'FLAG_OWN_REALTY': ['Y', 'N'],
}

CATEGORICAL_COLS = [
    'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE',
    'EMERGENCYSTATE_MODE',
]

CATEGORIES = [
    ['Cash loans', 'Revolving loans'],
    ['Children', 'Family', 'Group of people', 'Other_A', 'Other_B', 'Spouse, partner', 'Unaccompanied'],
    ['Businessman', 'Commercial associate', 'Maternity leave', 'Pensioner', 'State servant', 'Student', 'Unemployed', 'Working'],
    ['Academic degree', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Secondary / secondary special'],
    ['Civil marriage', 'Married', 'Separated', 'Single / not married', 'Unknown', 'Widow'],
    ['Co-op apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents'],
    ['Accountants', 'Cleaning staff', 'Cooking staff', 'Core staff', 'Drivers', 'HR staff', 'High skill tech staff', 'IT staff', 'Laborers', 'Low-skill Laborers', 'Managers', 'Medicine staff', 'Private service staff', 'Realty agents', 'Sales staff', 'Secretaries', 'Security staff', 'Waiters/barmen staff'],
    ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'],
    ['Advertising', 'Agriculture', 'Bank', 'Business Entity Type 1', 'Business Entity Type 2', 'Business Entity Type 3', 'Cleaning', 'Construction', 'Culture', 'Electricity', 'Emergency', 'Government', 'Hotel', 'Housing', 'Industry: type 1', 'Industry: type 10', 'Industry: type 11', 'Industry: type 12', 'Industry: type 13', 'Industry: type 2', 'Industry: type 3', 'Industry: type 4', 'Industry: type 5', 'Industry: type 6', 'Industry: type 7', 'Industry: type 8', 'Industry: type 9', 'Insurance', 'Kindergarten', 'Legal Services', 'Medicine', 'Military', 'Mobile', 'Other', 'Police', 'Postal', 'Realtor', 'Religion', 'Restaurant', 'School', 'Security', 'Security Ministries', 'Self-employed', 'Services', 'Telecom', 'Trade: type 1', 'Trade: type 2', 'Trade: type 3', 'Trade: type 4', 'Trade: type 5', 'Trade: type 6', 'Trade: type 7', 'Transport: type 1', 'Transport: type 2', 'Transport: type 3', 'Transport: type 4', 'University', 'XNA'],
    ['not specified', 'org spec account', 'reg oper account', 'reg oper spec account'],
    ['block of flats', 'specific housing', 'terraced house'],
    ['Block', 'Mixed', 'Monolithic', 'Others', 'Panel', 'Stone, brick', 'Wooden'],
    ['No', 'Yes'],
]


def load_data(execution_date):
    """Load data."""
    data_date = (execution_date - timedelta(days=1)).strftime("%Y-%m-%d")
    data_dir = BUCKET + "application/date_partition={}/".format(data_date)
    df = (
        pd.read_parquet(data_dir + "applications.gz.parquet")
        .query("CODE_GENDER != 'XNA'")   # Remove applications with XNA CODE_GENDER 
    )
    return df


def application(execution_date):
    """Preprocess applications."""
    raw_df = load_data(execution_date)
    
    # Swap target
    raw_df[TARGET] = 1 - raw_df[TARGET]
    
    # Binarize
    for col, val in BINARY_MAP.items():
        raw_df[col] = raw_df[col].apply(lambda x: 0 if x == val[0] else 1)
    
    # One-hot encoding of categorical features
    df, _ = onehot_enc(raw_df, CATEGORICAL_COLS, CATEGORIES)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # Some simple new features
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df
