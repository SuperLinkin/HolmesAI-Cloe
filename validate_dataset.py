"""
Dataset validation script to check data quality and statistics.
"""

import pandas as pd
import json


def validate_dataset(csv_path: str):
    """Validate the generated dataset."""
    print(f"Validating dataset: {csv_path}")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(csv_path)

    # Basic statistics
    print(f"\n1. Basic Statistics:")
    print(f"   Total transactions: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Check for missing values
    print(f"\n2. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found!")
    else:
        print(missing[missing > 0])

    # Check data types
    print(f"\n3. Data Types:")
    print(df.dtypes)

    # Validate required columns
    print(f"\n4. Column Validation:")
    required_cols = ['transaction_id', 'merchant_raw', 'amount', 'currency',
                     'timestamp', 'l1', 'l1_id', 'l2', 'l2_id', 'l3', 'l3_id']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"   ERROR: Missing columns: {missing_cols}")
    else:
        print("   All required columns present!")

    # Validate uniqueness
    print(f"\n5. Uniqueness Check:")
    print(f"   Unique transaction IDs: {df['transaction_id'].nunique()} / {len(df)}")
    duplicates = df['transaction_id'].duplicated().sum()
    if duplicates > 0:
        print(f"   WARNING: {duplicates} duplicate transaction IDs found!")
    else:
        print("   No duplicate transaction IDs!")

    # Category distribution
    print(f"\n6. Category Distribution:")
    print(f"   L1 categories: {df['l1'].nunique()}")
    print(f"   L2 categories: {df['l2'].nunique()}")
    print(f"   L3 categories: {df['l3'].nunique()}")

    # Top categories
    print(f"\n   Top 10 L1 categories:")
    for cat, count in df['l1'].value_counts().head(10).items():
        print(f"      {cat}: {count} ({count/len(df)*100:.1f}%)")

    # Amount validation
    print(f"\n7. Amount Validation:")
    print(f"   Min: ${df['amount'].min():.2f}")
    print(f"   Max: ${df['amount'].max():.2f}")
    print(f"   Mean: ${df['amount'].mean():.2f}")
    print(f"   Median: ${df['amount'].median():.2f}")
    negative_amounts = (df['amount'] <= 0).sum()
    if negative_amounts > 0:
        print(f"   WARNING: {negative_amounts} non-positive amounts found!")

    # Currency distribution
    print(f"\n8. Currency Distribution:")
    for curr, count in df['currency'].value_counts().items():
        print(f"   {curr}: {count} ({count/len(df)*100:.1f}%)")

    # Channel distribution
    print(f"\n9. Channel Distribution:")
    for channel, count in df['channel'].value_counts().items():
        print(f"   {channel}: {count} ({count/len(df)*100:.1f}%)")

    # MCC code validation
    print(f"\n10. MCC Code Validation:")
    mcc_present = df['mcc_code'].notna().sum()
    print(f"    Transactions with MCC: {mcc_present} ({mcc_present/len(df)*100:.1f}%)")

    # Date range
    print(f"\n11. Date Range:")
    df['timestamp_parsed'] = pd.to_datetime(df['timestamp'])
    print(f"    From: {df['timestamp_parsed'].min()}")
    print(f"    To: {df['timestamp_parsed'].max()}")
    print(f"    Duration: {(df['timestamp_parsed'].max() - df['timestamp_parsed'].min()).days} days")

    # Sample transactions
    print(f"\n12. Sample Transactions:")
    print(df.head(3).to_string())

    print("\n" + "=" * 70)
    print("Validation complete!")


def compare_splits(train_path: str, val_path: str, test_path: str):
    """Compare train/val/test splits."""
    print("\n" + "=" * 70)
    print("Comparing Train/Val/Test Splits")
    print("=" * 70)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    total = len(train) + len(val) + len(test)

    print(f"\n1. Split Sizes:")
    print(f"   Train: {len(train)} ({len(train)/total*100:.1f}%)")
    print(f"   Val: {len(val)} ({len(val)/total*100:.1f}%)")
    print(f"   Test: {len(test)} ({len(test)/total*100:.1f}%)")
    print(f"   Total: {total}")

    print(f"\n2. Category Distribution Comparison:")
    print(f"\n   L3 Categories:")
    print(f"   Train: {train['l3'].nunique()}")
    print(f"   Val: {val['l3'].nunique()}")
    print(f"   Test: {test['l3'].nunique()}")

    print(f"\n3. Check for Data Leakage:")
    train_ids = set(train['transaction_id'])
    val_ids = set(val['transaction_id'])
    test_ids = set(test['transaction_id'])

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("   ERROR: Data leakage detected!")
        if train_val_overlap:
            print(f"   Train-Val overlap: {len(train_val_overlap)} transactions")
        if train_test_overlap:
            print(f"   Train-Test overlap: {len(train_test_overlap)} transactions")
        if val_test_overlap:
            print(f"   Val-Test overlap: {len(val_test_overlap)} transactions")
    else:
        print("   No data leakage detected!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Validate each dataset
    datasets = [
        "data/synthetic_transactions_1k.csv",
        "data/synthetic_transactions_10k.csv",
        "data/synthetic_transactions_50k.csv"
    ]

    for dataset in datasets:
        try:
            validate_dataset(dataset)
            print("\n")
        except Exception as e:
            print(f"Error validating {dataset}: {e}")

    # Compare splits
    try:
        compare_splits("data/train.csv", "data/val.csv", "data/test.csv")
    except Exception as e:
        print(f"Error comparing splits: {e}")
