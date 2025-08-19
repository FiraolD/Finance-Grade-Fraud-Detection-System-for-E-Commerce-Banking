# src/features/engineer.py
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.helpers import time_diff_hours

def create_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df[f'{time_col}_hour'] = df[time_col].dt.hour
    df[f'{time_col}_dayofweek'] = df[time_col].dt.dayofweek
    df[f'{time_col}_is_weekend'] = (df[f'{time_col}_dayofweek'] >= 5).astype(int)
    return df

def create_time_since_signup(df: pd.DataFrame) -> pd.DataFrame:
    df['time_since_signup'] = df.apply(
        lambda row: time_diff_hours(row['signup_time'], row['purchase_time']),
        axis=1
    )
    return df

def create_transaction_velocity(df: pd.DataFrame, group_col: str, window: str = '24H') -> pd.DataFrame:
    df = df.sort_values('purchase_time')
    df[f'{group_col}_txn_count_24h'] = df.groupby(group_col).rolling(window, on='purchase_time').size().reset_index(level=0, drop=True)
    return df