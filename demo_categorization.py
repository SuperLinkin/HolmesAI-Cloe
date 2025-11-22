"""
Demo script to categorize transactions and display results.
"""

import pandas as pd
import requests
import json
from datetime import datetime
from collections import Counter
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_DATA_PATH = "data/synthetic_transactions_1k.csv"
SAMPLE_SIZE = 50  # Process first 50 transactions for demo

def load_test_data(sample_size=50):
    """Load test data from CSV."""
    df = pd.read_csv(TEST_DATA_PATH)
    print(f"[OK] Loaded {len(df)} transactions from dataset")

    # Take sample
    sample_df = df.head(sample_size)
    print(f"[OK] Processing {len(sample_df)} transactions for demo\n")

    return sample_df

def prepare_transactions(df):
    """Convert DataFrame to API format."""
    transactions = []

    for _, row in df.iterrows():
        txn = {
            "transaction_id": row['transaction_id'],
            "merchant_raw": row['merchant_raw'],
            "amount": float(row['amount']),
            "currency": row['currency'],
            "timestamp": row['timestamp'],
            "channel": row['channel'],
            "location": row['location'] if pd.notna(row['location']) else None,
            "mcc_code": str(int(row['mcc_code'])) if pd.notna(row['mcc_code']) else None
        }
        transactions.append(txn)

    return transactions

def categorize_transactions(transactions):
    """Send transactions to API for categorization."""
    url = f"{API_BASE_URL}/api/v1/categorize"

    print(f"[INFO] Sending {len(transactions)} transactions to API...")
    start_time = time.time()

    try:
        response = requests.post(
            url,
            json={"transactions": transactions},
            headers={"Content-Type": "application/json"}
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            results = response.json()
            print(f"[OK] Received {len(results['results'])} results")
            print(f"[OK] Processing time: {elapsed:.2f}s ({elapsed*1000/len(transactions):.2f}ms per transaction)\n")
            return results['results']
        else:
            print(f"[ERROR] API returned status {response.status_code}")
            print(f"Response: {response.text}")
            return []

    except Exception as e:
        print(f"[ERROR] Failed to connect to API: {e}")
        print(f"[INFO] Make sure the API is running at {API_BASE_URL}")
        return []

def display_results(results, original_df):
    """Display categorization results with statistics."""

    if not results:
        print("[ERROR] No results to display")
        return

    print("=" * 100)
    print("CATEGORIZATION RESULTS")
    print("=" * 100)

    # Sample results
    print("\nSAMPLE RESULTS (First 10 transactions):\n")
    print(f"{'ID':<20} {'Merchant':<25} {'Predicted L3':<35} {'Confidence':<12} {'Time'}")
    print("-" * 100)

    for i, result in enumerate(results[:10]):
        txn_id = result['transaction_id']
        merchant = original_df[original_df['transaction_id'] == txn_id]['merchant_raw'].values[0]
        merchant = (merchant[:22] + '...') if len(merchant) > 25 else merchant

        l3_pred = result['category']['l3']
        l3_pred = (l3_pred[:32] + '...') if len(l3_pred) > 35 else l3_pred

        confidence = result['confidence']
        conf_color = "✓" if confidence >= 0.85 else "~" if confidence >= 0.70 else "!"

        time_ms = result['processing_time_ms']

        print(f"{txn_id:<20} {merchant:<25} {l3_pred:<35} {conf_color} {confidence*100:>5.1f}%   {time_ms:>6.2f}ms")

    # Statistics
    print("\n" + "=" * 100)
    print("PERFORMANCE STATISTICS")
    print("=" * 100)

    # Confidence distribution
    confidences = [r['confidence'] for r in results]
    high_conf = sum(1 for c in confidences if c >= 0.85)
    med_conf = sum(1 for c in confidences if 0.70 <= c < 0.85)
    low_conf = sum(1 for c in confidences if c < 0.70)

    print(f"\nConfidence Distribution:")
    print(f"   High (>=85%):    {high_conf:>4} ({high_conf/len(results)*100:>5.1f}%)")
    print(f"   Medium (70-85%): {med_conf:>4} ({med_conf/len(results)*100:>5.1f}%)")
    print(f"   Low (<70%):      {low_conf:>4} ({low_conf/len(results)*100:>5.1f}%)")

    # Latency stats
    latencies = [r['processing_time_ms'] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    print(f"\nLatency:")
    print(f"   Average:  {avg_latency:>6.2f}ms")
    print(f"   Min:      {min_latency:>6.2f}ms")
    print(f"   Max:      {max_latency:>6.2f}ms")
    print(f"   Target:   <200ms {'[OK]' if avg_latency < 200 else '[FAIL]'}")

    # Category distribution
    l1_categories = [r['category']['l1'] for r in results]
    l1_counts = Counter(l1_categories)

    print(f"\nTop 5 L1 Categories:")
    for category, count in l1_counts.most_common(5):
        print(f"   {category:<20} {count:>4} ({count/len(results)*100:>5.1f}%)")

    # Accuracy (if ground truth available)
    if 'l3' in original_df.columns:
        print("\n" + "=" * 100)
        print("ACCURACY METRICS (vs Ground Truth)")
        print("=" * 100)

        correct_l1 = 0
        correct_l2 = 0
        correct_l3 = 0

        for result in results:
            txn_id = result['transaction_id']
            actual = original_df[original_df['transaction_id'] == txn_id].iloc[0]

            if result['category']['l1'] == actual['l1']:
                correct_l1 += 1
            if result['category']['l2'] == actual['l2']:
                correct_l2 += 1
            if result['category']['l3'] == actual['l3']:
                correct_l3 += 1

        l1_acc = correct_l1 / len(results) * 100
        l2_acc = correct_l2 / len(results) * 100
        l3_acc = correct_l3 / len(results) * 100

        print(f"\nL1 Accuracy: {l1_acc:>5.1f}% ({correct_l1}/{len(results)})")
        print(f"L2 Accuracy: {l2_acc:>5.1f}% ({correct_l2}/{len(results)})")
        print(f"L3 Accuracy: {l3_acc:>5.1f}% ({correct_l3}/{len(results)})")
        print(f"\nTarget L3:   >=90.0% {'[OK]' if l3_acc >= 90 else '[NEEDS IMPROVEMENT]'}")

    # Examples of categorizations
    print("\n" + "=" * 100)
    print("DETAILED EXAMPLES")
    print("=" * 100)

    for i in range(min(3, len(results))):
        result = results[i]
        txn_id = result['transaction_id']
        actual = original_df[original_df['transaction_id'] == txn_id].iloc[0]

        print(f"\nExample {i+1}:")
        print(f"  Transaction ID: {txn_id}")
        print(f"  Merchant:       {actual['merchant_raw']}")
        print(f"  Amount:         {actual['currency']} {actual['amount']}")
        print(f"  Channel:        {actual['channel']}")

        print(f"\n  Predicted:")
        print(f"    L1: {result['category']['l1']}")
        print(f"    L2: {result['category']['l2']}")
        print(f"    L3: {result['category']['l3']}")
        print(f"    Confidence: {result['confidence']*100:.1f}%")

        if 'l3' in actual:
            match = "[MATCH]" if result['category']['l3'] == actual['l3'] else "[DIFF]"
            print(f"\n  Actual: {actual['l3']} {match}")

        print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")

    print("\n" + "=" * 100)

def main():
    """Main demo function."""
    print("=" * 100)
    print("HOLMES AI - TRANSACTION CATEGORIZATION DEMO")
    print("=" * 100)
    print()

    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"[OK] API is healthy: {health['status']}")
            print(f"[OK] Models loaded: {health['models_loaded']}\n")
        else:
            print(f"[ERROR] API health check failed")
            return
    except Exception as e:
        print(f"[ERROR] Cannot connect to API at {API_BASE_URL}")
        print(f"[INFO] Please start the API with: uvicorn src.api.main:app --reload")
        return

    # Load data
    df = load_test_data(SAMPLE_SIZE)

    # Prepare transactions
    transactions = prepare_transactions(df)

    # Categorize
    results = categorize_transactions(transactions)

    # Display results
    if results:
        display_results(results, df)

    print("\n[INFO] Demo complete!")
    print(f"[INFO] View full results in the web dashboard at http://localhost:8080")
    print(f"[INFO] API documentation: {API_BASE_URL}/docs")

if __name__ == "__main__":
    main()
