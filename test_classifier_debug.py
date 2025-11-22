"""
Debug script to understand why L2/L3 accuracy is low.
"""

import pandas as pd
import numpy as np
from src.models import LightGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load validation data
df = pd.read_csv('data/synthetic_transactions_10k.csv')

# Take a small sample for analysis
sample = df.head(100)

print("="*80)
print("DEBUGGING L2/L3 LOW ACCURACY")
print("="*80)

# Check label distribution
print("\n1. LABEL DISTRIBUTION")
print("-"*80)
print(f"Unique L1 categories: {df['l1'].nunique()}")
print(f"Unique L2 categories: {df['l2'].nunique()}")
print(f"Unique L3 categories: {df['l3'].nunique()}")

print("\n2. LABEL FORMAT")
print("-"*80)
print("Sample L1 labels:", df['l1'].head(10).tolist())
print("Sample L2 labels:", df['l2'].head(10).tolist())
print("Sample L3 labels:", df['l3'].head(10).tolist())

# Check for hierarchical consistency
print("\n3. HIERARCHICAL CONSISTENCY CHECK")
print("-"*80)
for i in range(min(10, len(df))):
    row = df.iloc[i]
    l1 = row['l1']
    l2 = row['l2']
    l3 = row['l3']

    # Check if L2 starts with L1 content
    l2_consistent = l1 in l2 if isinstance(l2, str) and isinstance(l1, str) else False
    l3_consistent = l2 in l3 if isinstance(l3, str) and isinstance(l2, str) else False

    if not l2_consistent or not l3_consistent:
        print(f"Row {i}: L1={l1}, L2={l2}, L3={l3}")
        print(f"  L2 contains L1: {l2_consistent}, L3 contains L2: {l3_consistent}")

# Check L1 -> L2 -> L3 mappings
print("\n4. HIERARCHY MAPPINGS")
print("-"*80)
l1_to_l2 = df.groupby('l1')['l2'].nunique().to_dict()
l2_to_l3 = df.groupby('l2')['l3'].nunique().to_dict()

print("L2 categories per L1 category:")
for l1, count in sorted(l1_to_l2.items())[:5]:
    print(f"  {l1}: {count} L2 categories")

print("\nL3 categories per L2 category (sample):")
for l2, count in sorted(l2_to_l3.items())[:5]:
    print(f"  {l2}: {count} L3 categories")

# Check if models respect hierarchy
print("\n5. ANALYZING PREDICTED VS ACTUAL HIERARCHY")
print("-"*80)

# Load the trained classifier to check encoders
classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")
classifier.load_models("models/lightgbm")

print("\n6. CHECKING LABEL ENCODER CLASSES")
print("-"*80)
print(f"L1 encoder has {len(classifier.label_encoder_l1.classes_)} classes")
print(f"L2 encoder has {len(classifier.label_encoder_l2.classes_)} classes")
print(f"L3 encoder has {len(classifier.label_encoder_l3.classes_)} classes")

print("\nFirst 10 L1 classes:", classifier.label_encoder_l1.classes_[:10])
print("First 10 L2 classes:", classifier.label_encoder_l2.classes_[:10])
print("First 10 L3 classes:", classifier.label_encoder_l3.classes_[:10])
