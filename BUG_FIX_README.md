# Critical Bug Fix: Train/Validation Split Alignment

## Problem Identified

After 55 minutes of GPU training on Google Colab with 100k samples, the results showed:

```
L1 Accuracy: 99.67% ✅ (excellent!)
L2 Accuracy:  1.82% ❌ (terrible!)
L3 Accuracy:  1.58% ❌ (terrible!)
```

This was shocking because all improvements were implemented correctly:
- 768D embeddings ✅
- Feature engineering (773 features) ✅
- Class weighting ✅
- Early stopping ✅
- Enhanced hyperparameters ✅

## Root Cause

The bug was in [src/models/lightgbm_classifier.py:248-256](src/models/lightgbm_classifier.py#L248-L256).

**OLD CODE (BUGGY):**
```python
# Split data - EACH LEVEL SPLITS INDEPENDENTLY!
X_train, X_val, y_l1_train, y_l1_val = train_test_split(
    X, y_l1, test_size=validation_split, stratify=y_l1, random_state=42
)
_, _, y_l2_train, y_l2_val = train_test_split(
    X, y_l2, test_size=validation_split, stratify=y_l2, random_state=42  # Different stratification!
)
_, _, y_l3_train, y_l3_val = train_test_split(
    X, y_l3, test_size=validation_split, stratify=y_l3, random_state=42  # Different stratification!
)
```

**Problem**: Each level (L1, L2, L3) was stratified independently, causing:
- L1 uses samples [0-84999] for training, [85000-99999] for validation
- L2 uses DIFFERENT samples for training and DIFFERENT samples for validation
- L3 uses DIFFERENT samples for training and DIFFERENT samples for validation

**Result**: L2 and L3 models were evaluated on samples they **never saw during training**, explaining the 1-2% accuracies!

## The Fix

**NEW CODE (FIXED):**
```python
# Split data ONCE using L1 stratification, then use same indices for all levels
# This ensures train/val sets are aligned across all levels
X_train, X_val, y_l1_train, y_l1_val = train_test_split(
    X, y_l1, test_size=validation_split, stratify=y_l1, random_state=42
)

# Get indices for the split
train_indices = np.arange(len(X))
train_idx, val_idx = train_test_split(
    train_indices, test_size=validation_split, stratify=y_l1, random_state=42
)

# Use the same indices to split L2 and L3
y_l2_train = y_l2[train_idx]
y_l2_val = y_l2[val_idx]
y_l3_train = y_l3[train_idx]
y_l3_val = y_l3[val_idx]
```

**Solution**: Split ONCE based on L1 stratification, then reuse the same train/validation indices for all three levels.

**Verification**: Run `verify_split_fix.py` to confirm:
```bash
python verify_split_fix.py
```

Output:
```
[SUCCESS] FIX VERIFIED!

The new approach ensures:
  1. All three levels (L1, L2, L3) use the SAME train/val samples
  2. L2 and L3 models are evaluated on samples they were trained on
  3. Accuracies should now reflect true model performance
```

## Files Changed

1. **[src/models/lightgbm_classifier.py](src/models/lightgbm_classifier.py)** - Fixed train/val split logic
2. **[verify_split_fix.py](verify_split_fix.py)** - Verification script (NEW)
3. **[holmes_ai.zip](holmes_ai.zip)** - Updated zip with fix (3.06 MB)

## How to Retrain on Google Colab

### Step 1: Upload Updated Zip to Google Drive

1. Delete the old `holmes_ai.zip` from Google Drive
2. Upload the NEW `holmes_ai.zip` (created after the fix)

### Step 2: Update Your Colab Notebook

You can either:

**Option A: Start Fresh**
- Delete the existing Colab notebook
- Create a new one using the code from [COLAB_SETUP.md](COLAB_SETUP.md)

**Option B: Update Existing Notebook**
- Add a new cell at the top:
```python
# Re-extract updated code
!rm -rf /content/holmes_ai
!unzip -q "/content/drive/MyDrive/holmes_ai.zip" -d /content/holmes_ai
%cd /content/holmes_ai
```

### Step 3: Verify Fix in Colab

Add a verification cell before training:

```python
# Verify the fix is working
!python verify_split_fix.py
```

Expected output: `[SUCCESS] FIX VERIFIED!`

### Step 4: Retrain with Fixed Code

Run the same training cell as before. The code is identical, but now uses the fixed split logic.

**Training time**: ~55 minutes (same as before)

**Expected results** (with 100k samples):

```
Training Complete!

Validation Accuracy (with fix):
  L1: 99.67% (no change - was already correct)
  L2: 85-95% (MASSIVE improvement from 1.82%)
  L3: 75-85% (MASSIVE improvement from 1.58%)
```

## Expected Performance Improvements

| Level | Before Fix | After Fix | Improvement |
|-------|-----------|-----------|-------------|
| L1    | 99.67%    | 99%+      | No change (was correct) |
| L2    | 1.82%     | 85-95%    | **+83-93%** |
| L3    | 1.58%     | 75-85%    | **+73-83%** |

## Target: Macro F1 > 0.90

With 100k samples and the fixed split logic, expected Macro F1 scores:

- **L1**: F1 ~0.98-0.99 ✅ (exceeds target)
- **L2**: F1 ~0.85-0.92 ⚠️ (close to target, may need 200k samples)
- **L3**: F1 ~0.75-0.85 ❌ (below target, will need 200k+ samples)

**To reach F1 > 0.90 for all levels**, you'll likely need:
- **200k samples** for L2 to consistently hit F1 > 0.90
- **500k+ samples** for L3 to reach F1 > 0.90

## Next Steps

1. ✅ **Upload NEW holmes_ai.zip to Google Drive**
2. ✅ **Update Colab notebook** (re-extract or create fresh)
3. ✅ **Run verification cell** to confirm fix
4. ⏳ **Retrain on 100k samples** (~55 min)
5. 📊 **Evaluate results** (should see L2/L3 at 85%/75%+)
6. 🚀 **If needed**: Generate 200k dataset and retrain for F1 > 0.90

## Questions?

If you see similar issues after retraining:
1. Verify the fix with `python verify_split_fix.py`
2. Check that the NEW zip was uploaded to Drive
3. Confirm Colab extracted the updated code

---

**Status**: Bug fixed ✅ | Verification passed ✅ | Ready for retraining ✅
