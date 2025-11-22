# Google Colab Training Setup Guide

This guide explains how to train Holmes AI models on Google Colab Pro for 10-15x faster training with GPU acceleration.

---

## Why Google Colab Pro?

**Current Performance (Laptop CPU):**
- 50k samples: ~30-45 minutes
- 100k samples: ~60-90 minutes
- 200k samples: ~2-3 hours

**Expected Performance (Colab Pro GPU):**
- 50k samples: ~3-5 minutes
- 100k samples: ~6-10 minutes
- 200k samples: ~12-20 minutes

**Speed-up: 10-15x faster** ⚡

---

## Prerequisites

1. **Google Colab Pro Subscription** ($9.99/month)
   - Tesla T4 or V100 GPU
   - 25GB RAM
   - Longer runtime (24 hours vs 12 hours)

2. **Google Drive** (for dataset storage)

---

## Setup Instructions

### 1. Upload Project to Google Drive

```bash
# On your local machine
# Zip the project (excluding models and large files)
zip -r holmes_ai.zip . -x "models/*" "data/*.csv" "venv/*" ".git/*"
```

Then upload `holmes_ai.zip` to your Google Drive.

### 2. Generate Large Training Dataset

```bash
# Generate 100k samples locally
python generate_dataset.py --samples 100000 --output data/synthetic_transactions_100k.csv

# Or 200k samples
python generate_dataset.py --samples 200000 --output data/synthetic_transactions_200k.csv
```

Upload the CSV to Google Drive.

### 3. Create Colab Notebook

Create a new notebook in Google Colab: `Holmes_AI_Training.ipynb`

---

## Colab Notebook Code

### Cell 1: Setup Environment

```python
# Check GPU availability
!nvidia-smi

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract project
!unzip -q "/content/drive/MyDrive/holmes_ai.zip" -d /content/holmes_ai
%cd /content/holmes_ai

# Install dependencies
!pip install -q sentence-transformers lightgbm scikit-learn pandas numpy
```

### Cell 2: Verify GPU

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Cell 3: Train with All Improvements

```python
import sys
sys.path.append('/content/holmes_ai')

from src.data_ingestion import DataIngestion
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier
import pandas as pd
import numpy as np

# Load data from Google Drive
DATA_PATH = "/content/drive/MyDrive/synthetic_transactions_100k.csv"

print("=" * 80)
print("HOLMES AI - GPU TRAINING")
print("=" * 80)

# Step 1: Load data
print("\n[1/7] Loading data...")
ingestion = DataIngestion()
df = pd.read_csv(DATA_PATH)
normalized = ingestion.ingest_pipeline(DATA_PATH)

transactions = []
for i, txn in enumerate(normalized):
    txn_dict = txn.model_dump()
    row = df.iloc[i]
    txn_dict['l1'] = row['l1']
    txn_dict['l2'] = row['l2']
    txn_dict['l3'] = row['l3']
    transactions.append(txn_dict)

print(f"[OK] Loaded {len(transactions):,} transactions")

# Step 2: Preprocess
print("\n[2/7] Preprocessing...")
preprocessor = TransactionPreprocessor()
preprocessed = preprocessor.preprocess_batch(transactions)
print(f"[OK] Preprocessed {len(preprocessed):,} transactions")

# Step 3: Encode with Sentence-BERT (GPU-accelerated)
print("\n[3/7] Encoding with Sentence-BERT (GPU)...")
encoder = SentenceBERTEncoder()  # Will use GPU automatically
embeddings = encoder.encode_transactions(
    preprocessed,
    text_field='merchant_cleaned',
    batch_size=64  # Larger batch size for GPU
)
print(f"[OK] Generated {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]}D)")

# Step 4: Prepare labels
print("\n[4/7] Preparing labels...")
classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")

y_l1 = classifier.prepare_labels(preprocessed, level='l1')
y_l2 = classifier.prepare_labels(preprocessed, level='l2')
y_l3 = classifier.prepare_labels(preprocessed, level='l3')

print(f"[OK] L1: {len(np.unique(y_l1))} classes")
print(f"[OK] L2: {len(np.unique(y_l2))} classes")
print(f"[OK] L3: {len(np.unique(y_l3))} classes")

# Step 5: Build hierarchy maps
print("\n[5/7] Building hierarchy maps...")
classifier.build_hierarchy_maps(preprocessed)

# Step 6: Prepare features
print("\n[6/7] Preparing features...")
X = classifier.prepare_features(
    embeddings,
    preprocessed,
    include_enrichment=True
)
print(f"[OK] Feature matrix: {X.shape}")

# Step 7: Train with all improvements
print("\n[7/7] Training with all improvements...")
print("\nTraining configuration:")
print(f"  ├─ Samples: {len(transactions):,}")
print(f"  ├─ Features: {X.shape[1]} (768 embeddings + 5 engineered)")
print(f"  ├─ Embedding model: all-mpnet-base-v2 (768D)")
print(f"  ├─ Class weighting: Enabled")
print(f"  ├─ Early stopping: 50 rounds")
print(f"  ├─ Max boosting rounds: 500")
print(f"  └─ Device: GPU")
print()

import time
start_time = time.time()

scores = classifier.train(
    X, y_l1, y_l2, y_l3,
    validation_split=0.15,
    num_boost_round=500,
    early_stopping_rounds=50,
    use_class_weight=True
)

training_time = time.time() - start_time

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nTraining time: {training_time/60:.1f} minutes")
print(f"\nValidation Accuracy (without hierarchy):")
print(f"  L1: {scores['l1_accuracy']:.4f} ({scores['l1_accuracy']*100:.2f}%)")
print(f"  L2: {scores['l2_accuracy']:.4f} ({scores['l2_accuracy']*100:.2f}%)")
print(f"  L3: {scores['l3_accuracy']:.4f} ({scores['l3_accuracy']*100:.2f}%)")

# Save models to Google Drive
print("\n[SAVING] Saving models to Google Drive...")
save_path = "/content/drive/MyDrive/holmes_models_improved"
encoder.save_model(f"{save_path}/sentence_bert")
classifier.save_models(f"{save_path}/lightgbm")
print(f"[OK] Models saved to: {save_path}")
```

### Cell 4: Test Hierarchical Accuracy

```python
# Test with hierarchical filtering
print("\n" + "=" * 80)
print("TESTING HIERARCHICAL PREDICTION")
print("=" * 80)

# Make predictions WITHOUT hierarchy
predictions_no_hier = classifier.predict(X, use_hierarchy=False)

# Make predictions WITH hierarchy
predictions_hier = classifier.predict(X, use_hierarchy=True)

# Get ground truth
from sklearn.preprocessing import LabelEncoder
le_l2 = LabelEncoder()
le_l3 = LabelEncoder()

y_true_l2 = le_l2.fit_transform([t['l2'] for t in preprocessed])
y_true_l3 = le_l3.fit_transform([t['l3'] for t in preprocessed])

# Compare accuracies
pred_l2_no_hier = le_l2.transform(predictions_no_hier['l2'])
pred_l2_hier = le_l2.transform(predictions_hier['l2'])
pred_l3_no_hier = le_l3.transform(predictions_no_hier['l3'])
pred_l3_hier = le_l3.transform(predictions_hier['l3'])

acc_l2_no_hier = np.mean(pred_l2_no_hier == y_true_l2)
acc_l2_hier = np.mean(pred_l2_hier == y_true_l2)
acc_l3_no_hier = np.mean(pred_l3_no_hier == y_true_l3)
acc_l3_hier = np.mean(pred_l3_hier == y_true_l3)

print("\nComparison:")
print(f"\nL2 Accuracy:")
print(f"  Without hierarchy: {acc_l2_no_hier:.4f} ({acc_l2_no_hier*100:.2f}%)")
print(f"  With hierarchy:    {acc_l2_hier:.4f} ({acc_l2_hier*100:.2f}%)")
print(f"  Improvement:       +{(acc_l2_hier - acc_l2_no_hier)*100:.2f}%")

print(f"\nL3 Accuracy:")
print(f"  Without hierarchy: {acc_l3_no_hier:.4f} ({acc_l3_no_hier*100:.2f}%)")
print(f"  With hierarchy:    {acc_l3_hier:.4f} ({acc_l3_hier*100:.2f}%)")
print(f"  Improvement:       +{(acc_l3_hier - acc_l3_no_hier)*100:.2f}%")

# Check hierarchy violations
violations = 0
for i in range(len(predictions_hier['l1'])):
    l1_pred = predictions_hier['l1'][i]
    l2_pred = predictions_hier['l2'][i]

    valid_l2s = classifier.l1_to_l2_map.get(l1_pred, [])
    if valid_l2s and l2_pred not in valid_l2s:
        violations += 1

violation_rate = violations / len(predictions_hier['l1'])
print(f"\nHierarchy violations: {violations}/{len(predictions_hier['l1'])} ({violation_rate*100:.2f}%)")
```

### Cell 5: Calculate Macro F1 Scores

```python
from sklearn.metrics import f1_score, classification_report

print("\n" + "=" * 80)
print("MACRO F1 SCORES")
print("=" * 80)

# L1 F1
f1_l1 = f1_score(y_l1_val, pred_l1_val, average='macro')
print(f"\nL1 Macro F1: {f1_l1:.4f}")

# L2 F1
f1_l2 = f1_score(y_true_l2, pred_l2_hier, average='macro')
print(f"L2 Macro F1: {f1_l2:.4f}")

# L3 F1
f1_l3 = f1_score(y_true_l3, pred_l3_hier, average='macro')
print(f"L3 Macro F1: {f1_l3:.4f}")

print("\n" + "=" * 80)
print("TARGET: Macro F1 > 0.90 for all levels")
print("=" * 80)
```

---

## Expected Results with 100k Samples

```
Training time: ~8-12 minutes (vs 60-90 min on CPU)

Validation Accuracy:
  L1: 0.8800-0.9200 (88-92%)
  L2: 0.6500-0.7500 (65-75%)
  L3: 0.4800-0.6000 (48-60%)

With Hierarchical Filtering:
  L2: +25-30% improvement
  L3: +15-20% improvement
```

---

## Tips for Best Results

1. **Use Larger Batch Size on GPU**
   - CPU: batch_size=32
   - GPU: batch_size=64 or 128

2. **Monitor GPU Memory**
   ```python
   !nvidia-smi -l 1  # Monitor every second
   ```

3. **Save Checkpoints**
   - Save models every 100 rounds
   - In case of runtime disconnect

4. **Increase Dataset Size**
   - 200k samples → better L2/L3 accuracy
   - 500k samples → approach F1 > 0.9

---

## Troubleshooting

### GPU Not Available
```python
# Force GPU device
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### Out of Memory
- Reduce batch size: `batch_size=32`
- Reduce num_boost_round: `400 instead of 500`
- Use `!nvidia-smi` to check memory

### Runtime Disconnect
- Colab Pro gives 24-hour runtimes
- Save models frequently
- Use `keep_alive` extension

---

## Next Steps After Training

1. **Download Trained Models**
   - From Google Drive to local machine
   - Test with real transactions

2. **Deploy to Production**
   - Update API with new models
   - Test latency (<200ms target)

3. **Monitor Performance**
   - Track F1 scores on real data
   - Retrain monthly with new data

---

**Ready to train on Colab Pro? Let's achieve F1 > 0.9!** 🚀
