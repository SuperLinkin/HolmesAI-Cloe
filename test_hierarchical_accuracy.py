"""
Test script to evaluate the improvement from hierarchical prediction.
"""

import pandas as pd
import numpy as np
from src.data_ingestion import DataIngestion
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier

print("="*80)
print("TESTING HIERARCHICAL PREDICTION IMPROVEMENT")
print("="*80)

# Load test data (using a subset)
df = pd.read_csv('data/synthetic_transactions_10k.csv')
test_df = df.sample(n=1000, random_state=42)

# Process transactions
print("\n1. Loading and preprocessing test data...")
ingestion = DataIngestion()
preprocessor = TransactionPreprocessor()
encoder = SentenceBERTEncoder()
classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")

# Convert to transaction objects
transactions = []
for _, row in test_df.iterrows():
    txn = {
        'transaction_id': row['transaction_id'],
        'merchant_raw': row['merchant_raw'],
        'amount': row['amount'],
        'currency': row['currency'],
        'timestamp': row['timestamp'],
        'channel': row['channel'],
        'location': row.get('location') if pd.notna(row.get('location')) else None,
        'mcc_code': row.get('mcc_code'),
        'l1': row['l1'],
        'l2': row['l2'],
        'l3': row['l3']
    }
    transactions.append(txn)

# Preprocess
preprocessed = preprocessor.preprocess_batch(transactions)

# Encode
print("\n2. Encoding transactions...")
encoder.load_model("models/sentence_bert")
embeddings = encoder.encode_transactions(preprocessed, batch_size=32)

# Load classifier
print("\n3. Loading trained classifier...")
classifier.load_models("models/lightgbm")

# Prepare features
features = classifier.prepare_features(embeddings, preprocessed)

# Get true labels
y_true_l1 = [t['l1'] for t in transactions]
y_true_l2 = [t['l2'] for t in transactions]
y_true_l3 = [t['l3'] for t in transactions]

print("\n4. Evaluating WITHOUT hierarchical filtering...")
predictions_no_hierarchy = classifier.predict(features, return_proba=False, use_hierarchy=False)

acc_l1_no_hier = np.mean(predictions_no_hierarchy['l1'] == y_true_l1)
acc_l2_no_hier = np.mean(predictions_no_hierarchy['l2'] == y_true_l2)
acc_l3_no_hier = np.mean(predictions_no_hierarchy['l3'] == y_true_l3)

print(f"  L1 Accuracy (no hierarchy): {acc_l1_no_hier:.4f}")
print(f"  L2 Accuracy (no hierarchy): {acc_l2_no_hier:.4f}")
print(f"  L3 Accuracy (no hierarchy): {acc_l3_no_hier:.4f}")

print("\n5. Evaluating WITH hierarchical filtering...")
predictions_with_hierarchy = classifier.predict(features, return_proba=False, use_hierarchy=True)

acc_l1_hier = np.mean(predictions_with_hierarchy['l1'] == y_true_l1)
acc_l2_hier = np.mean(predictions_with_hierarchy['l2'] == y_true_l2)
acc_l3_hier = np.mean(predictions_with_hierarchy['l3'] == y_true_l3)

print(f"  L1 Accuracy (with hierarchy): {acc_l1_hier:.4f}")
print(f"  L2 Accuracy (with hierarchy): {acc_l2_hier:.4f}")
print(f"  L3 Accuracy (with hierarchy): {acc_l3_hier:.4f}")

print("\n6. IMPROVEMENT FROM HIERARCHICAL FILTERING")
print("="*80)
print(f"  L1: {acc_l1_no_hier:.4f} -> {acc_l1_hier:.4f} (change: {(acc_l1_hier - acc_l1_no_hier):.4f})")
print(f"  L2: {acc_l2_no_hier:.4f} -> {acc_l2_hier:.4f} (change: {(acc_l2_hier - acc_l2_no_hier):.4f})")
print(f"  L3: {acc_l3_no_hier:.4f} -> {acc_l3_hier:.4f} (change: {(acc_l3_hier - acc_l3_no_hier):.4f})")

# Check hierarchy violations
print("\n7. CHECKING HIERARCHY VIOLATIONS")
print("="*80)

violations_before = 0
violations_after = 0

for i in range(len(transactions)):
    true_l1, true_l2, true_l3 = y_true_l1[i], y_true_l2[i], y_true_l3[i]

    # Check before (no hierarchy)
    pred_l1_before = predictions_no_hierarchy['l1'][i]
    pred_l2_before = predictions_no_hierarchy['l2'][i]
    pred_l3_before = predictions_no_hierarchy['l3'][i]

    # Check if predicted L2 belongs to predicted L1
    if pred_l1_before in classifier.l1_to_l2_map:
        valid_l2s = classifier.l1_to_l2_map[pred_l1_before]
        if pred_l2_before not in valid_l2s:
            violations_before += 1

    # Check after (with hierarchy)
    pred_l1_after = predictions_with_hierarchy['l1'][i]
    pred_l2_after = predictions_with_hierarchy['l2'][i]
    pred_l3_after = predictions_with_hierarchy['l3'][i]

    if pred_l1_after in classifier.l1_to_l2_map:
        valid_l2s = classifier.l1_to_l2_map[pred_l1_after]
        if pred_l2_after not in valid_l2s:
            violations_after += 1

print(f"Hierarchy violations (L1->L2):")
print(f"  Before hierarchical filtering: {violations_before}/{len(transactions)} ({100*violations_before/len(transactions):.2f}%)")
print(f"  After hierarchical filtering: {violations_after}/{len(transactions)} ({100*violations_after/len(transactions):.2f}%)")

print("\n8. SAMPLE PREDICTIONS")
print("="*80)
for i in range(min(5, len(transactions))):
    print(f"\nTransaction {i+1}: {transactions[i]['merchant_raw']}")
    print(f"  True:      L1={y_true_l1[i][:20]:<20} L2={y_true_l2[i][:30]:<30} L3={y_true_l3[i][:40]}")
    print(f"  Predicted: L1={predictions_with_hierarchy['l1'][i][:20]:<20} L2={predictions_with_hierarchy['l2'][i][:30]:<30} L3={predictions_with_hierarchy['l3'][i][:40]}")

    # Check if correct
    l1_correct = predictions_with_hierarchy['l1'][i] == y_true_l1[i]
    l2_correct = predictions_with_hierarchy['l2'][i] == y_true_l2[i]
    l3_correct = predictions_with_hierarchy['l3'][i] == y_true_l3[i]
    print(f"  Correct:   L1={'YES' if l1_correct else 'NO':<20}  L2={'YES' if l2_correct else 'NO':<30}  L3={'YES' if l3_correct else 'NO'}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
