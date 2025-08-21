# ...existing code...
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.helpers import ip_to_int
import pandas as pd




import socket
import struct

def ip_to_int(ip_str: str) -> int:
    """
    Convert dotted IP string (e.g. '192.168.0.1') into integer.
    """
    try:
        return struct.unpack("!I", socket.inet_aton(ip_str))[0]
    except Exception:
        return 0

def float_to_ip(ip_num: float) -> str:
    """
    Convert float/integer IP to dotted string.
    Example: 3232235777 → '192.168.0.1'
    """
    try:
        return socket.inet_ntoa(struct.pack("!I", int(ip_num)))
    except Exception:
        return "0.0.0.0"


def load_fraud_data(path: str) -> pd.DataFrame:
    print("🔍 Loading fraud transaction data...")
    df = pd.read_csv(path)
    print(f"✅ Loaded fraud data: shape = {df.shape}")

    # Fix corrupted IP column
    print("🔧 Fixing corrupted IP addresses...")
    df['ip_address'] = df['ip_address'].astype(float).astype(int).apply(float_to_ip)

    
    # Show sample
    print("📌 Fixed IP samples:")
    print(df['ip_address'].head(10).tolist())

    # Convert timestamps
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    df['ip_int'] = df['ip_address'].apply(ip_to_int)
    return df


def load_credit_data(path: str) -> pd.DataFrame:
    """
    Load the credit card transaction data.
    """
    print("🔍 Loading credit card transaction data...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Credit card data file not found at: {path}")
    
    df = pd.read_csv(path)
    print(f"✅ Loaded credit card data: shape = {df.shape}")

    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️  Found {missing} missing values in credit card data.")
    else:
        print("✅ No missing values in credit card data.")

    return df


def load_ip_country_data(path: str) -> pd.DataFrame:
    """
    Load the IP address to country mapping data.
    """
    print("🔍 Loading IP-to-country mapping data...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"IP country mapping file not found at: {path}")
    
    df = pd.read_csv(path)
    print(f"✅ Loaded IP-to-country data: shape = {df.shape}")

    # Convert IP bounds to integers
    print("⏳ Converting 'lower_bound_ip_address' and 'upper_bound_ip_address' to integers...")
    df['lower_bound_ip_address'] = df['lower_bound_ip_address'].astype(int)
    df['upper_bound_ip_address'] = df['upper_bound_ip_address'].astype(int)
    print("✅ IP bounds converted to integers.")

    return df


def merge_ip_with_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fraud data with country information using IP address range mapping.
    Uses merge_asof on lower_bound and then filters by upper_bound to enforce the interval.
    """
    print("🌐 Starting IP-to-country merge using range-based lookup...")

    # Convert IP addresses to integers
    print("⏳ Converting fraud data IP addresses to integers...")
    fraud_df = fraud_df.copy()
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    print(f"✅ Created 'ip_int' column. Sample: {fraud_df['ip_int'].head(3).tolist()}")

    # Sort data for merge_asof
    print("⏳ Sorting fraud and IP-country data for efficient merge...")
    fraud_df_sorted = fraud_df.sort_values('ip_int').copy()
    ip_df_sorted = ip_df.sort_values('lower_bound_ip_address').copy()

    print(f"✅ Fraud data sorted by ip_int ({len(fraud_df_sorted)} rows).")
    print(f"✅ IP-country data sorted by lower_bound_ip_address ({len(ip_df_sorted)} rows).")

    # Perform the merge_asof (matches nearest lower_bound <= ip_int)
    print("🔗 Performing pd.merge_asof() for IP range matching...")
    merged = pd.merge_asof(
        fraud_df_sorted,
        ip_df_sorted[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        allow_exact_matches=True
    )

    # Keep only rows where ip_int <= upper_bound_ip_address (enforce interval)
    merged = merged[merged['ip_int'] <= merged['upper_bound_ip_address']].copy()

    # Rename and report
    merged.rename(columns={'country': 'transaction_country'}, inplace=True)
    print("✅ IP-to-country merge completed.")
    
    # Report coverage
    matched = merged['transaction_country'].notna().sum()
    total = len(merged)
    print(f"📊 IP-to-country match rate: {matched}/{total} ({(matched/total) if total else 0:.2%})")

    if matched < total:
        unmatched = total - matched
        print(f"⚠️  {unmatched} transactions could not be mapped to a country (likely private/local IPs).")

    return merged


if __name__ == "__main__":
    # Run the loading + merge pipeline from project root
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    fraud_path = os.path.join(base, 'Data', 'Fraud_Data.csv')
    ip_path = os.path.join(base, 'Data', 'IpAddress_to_Country.csv')
    out_path = os.path.join(base, 'Data', 'merged_data.csv')

    fraud_df = load_fraud_data(fraud_path)
    ip_df = load_ip_country_data(ip_path)
    fraud_enriched = merge_ip_with_country(fraud_df, ip_df)
    fraud_enriched.to_csv(out_path, index=False)
    print(f"✅ Enriched fraud data saved to: {out_path}")
# ...existing code...