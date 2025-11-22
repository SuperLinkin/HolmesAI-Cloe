"""
Analyze the root cause of low L2/L3 accuracy.
"""

import pandas as pd
import numpy as np
from src.models import LightGBMClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/synthetic_transactions_10k.csv')

# Load the trained classifier
classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")
classifier.load_models("models/lightgbm")

print("="*80)
print("ROOT CAUSE ANALYSIS: LOW L2/L3 ACCURACY")
print("="*80)

# Create label mappings from the data
l1_to_l2_map = {}
l2_to_l3_map = {}

for _, row in df.iterrows():
    l1, l2, l3 = row['l1'], row['l2'], row['l3']

    if l1 not in l1_to_l2_map:
        l1_to_l2_map[l1] = set()
    l1_to_l2_map[l1].add(l2)

    if l2 not in l2_to_l3_map:
        l2_to_l3_map[l2] = set()
    l2_to_l3_map[l2].add(l3)

print("\n1. HIERARCHICAL STRUCTURE FROM DATA")
print("-"*80)
print(f"Total L1 categories: {len(l1_to_l2_map)}")
print(f"Average L2 per L1: {np.mean([len(v) for v in l1_to_l2_map.values()]):.2f}")
print(f"Average L3 per L2: {np.mean([len(v) for v in l2_to_l3_map.values()]):.2f}")

print("\n2. KEY ISSUE: INDEPENDENT CLASSIFIERS")
print("-"*80)
print("The current implementation trains L1, L2, and L3 classifiers INDEPENDENTLY.")
print("This means:")
print("  [X] L2 classifier doesn't know which L1 was predicted")
print("  [X] L3 classifier doesn't know which L1/L2 was predicted")
print("  [X] Models can predict invalid hierarchies (e.g., L1=Travel, L2=Dining-FastFood)")

print("\n3. EXAMPLE OF THE PROBLEM")
print("-"*80)

# Simulate what happens with independent classifiers
sample_row = df.iloc[0]
print(f"Actual hierarchy:")
print(f"  L1: {sample_row['l1']}")
print(f"  L2: {sample_row['l2']}")
print(f"  L3: {sample_row['l3']}")

print(f"\nWith independent classifiers, the model might predict:")
print(f"  L1: {sample_row['l1']} [OK] (correct)")
print(f"  L2: Education - Courses [WRONG] (belongs to different L1)")
print(f"  L3: Shopping - Electronics - Laptop [WRONG] (belongs to different L1/L2)")

print("\n4. WHY L1 ACCURACY IS HIGHER")
print("-"*80)
print("L1 has only 15 classes and broader categories (Travel, Dining, etc.)")
print("The semantic embeddings can distinguish high-level categories well.")
print("")
print("L2 and L3 are much more granular:")
print(f"  - L2: 42 classes (must match specific L1 parent)")
print(f"  - L3: 59 classes (must match specific L1+L2 parent)")
print("")
print("Without hierarchical constraints, the classifier treats all 42 L2")
print("categories as independent choices, ignoring the L1 prediction.")

print("\n5. SOLUTION: HIERARCHICAL CLASSIFICATION")
print("-"*80)
print("We need to implement one of these approaches:")
print("")
print("Option 1: CONDITIONAL PREDICTION (Recommended)")
print("  1. Predict L1 first")
print("  2. Filter L2 candidates to only those under predicted L1")
print("  3. Predict L2 from filtered set")
print("  4. Filter L3 candidates to only those under predicted L2")
print("  5. Predict L3 from filtered set")
print("")
print("Option 2: HIERARCHICAL FEATURES")
print("  - Include L1 prediction as a feature for L2 classifier")
print("  - Include L1+L2 predictions as features for L3 classifier")
print("")
print("Option 3: LOCAL CLASSIFIERS")
print("  - Train separate L2 classifier for each L1 category")
print("  - Train separate L3 classifier for each L2 category")
print("  - This creates many small models but respects hierarchy")

print("\n6. VALIDATION SPLIT ISSUE")
print("-"*80)
# Check if train/val split is consistent
X = np.random.rand(len(df), 384)  # Mock features
y_l1 = classifier.label_encoder_l1.transform(df['l1'])
y_l2 = classifier.label_encoder_l2.transform(df['l2'])
y_l3 = classifier.label_encoder_l3.transform(df['l3'])

# These splits use the SAME random_state=42
X_train1, X_val1, y_l1_train, y_l1_val = train_test_split(X, y_l1, test_size=0.15, stratify=y_l1, random_state=42)
X_train2, X_val2, y_l2_train, y_l2_val = train_test_split(X, y_l2, test_size=0.15, stratify=y_l2, random_state=42)

# Check if samples are aligned
samples_aligned = np.array_equal(X_val1, X_val2)
print(f"Validation samples aligned across L1/L2/L3: {samples_aligned}")

if not samples_aligned:
    print("[!] WARNING: Different stratification for each level may cause misalignment")
else:
    print("[OK] Validation splits are consistent")
