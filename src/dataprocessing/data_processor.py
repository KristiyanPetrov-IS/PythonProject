"""File used to process the load, analyze and clean the data"""
import pandas as pd
from src.defines import SEPARATOR_ONE, SEPARATOR_TWO
from src.defines import INDEPENDENT_FIELD, INDEPENDENT_FIELD_LEN

def load_data(filepath):
    """Create a dataframe with data in csv file"""
    return pd.read_csv(filepath)

def analyze_df(df, check_null = False):
    """Print analytics for the data"""
    print(SEPARATOR_ONE)
    print(df.describe())
    print(SEPARATOR_TWO)
    print(df.info())
    if check_null:
        print(SEPARATOR_TWO)
        print("Брой null стойности по полета:")
        print(df.isnull().sum())
        print("Брой дуплицирани редове:")
        print(df.duplicated().sum())
        print(df.head())
    print(SEPARATOR_ONE)

def handle_extremal_values(df, column):
    """Remove extremal values"""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def prepare_df(df):
    """clean and analyze the data"""
    df[INDEPENDENT_FIELD_LEN] = df[INDEPENDENT_FIELD].apply(len)
    analyze_df(df)
    df = handle_extremal_values(df, column=INDEPENDENT_FIELD_LEN)
    analyze_df(df, check_null=True)
    return df
