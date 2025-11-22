# Holmes AI - Explainability Guide

## Overview

Holmes AI now includes **enhanced explainability** beyond simple confidence scores, providing comprehensive insights into why predictions were made.

---

## ✅ Explainability Features

### 1. **SHAP Feature Importance Analysis**

Uses SHAP (SHapley Additive exPlanations) to determine which features contributed most to each prediction.

**Benefits:**
- Identifies top contributing features (semantic embeddings vs engineered features)
- Quantifies impact of each feature on the prediction
- Provides model-agnostic explanations

**Usage:**
```bash
# Generate feature importance report
python explainability.py --mode batch --data data/test.csv --output explainability_results
```

**Output:**
- `feature_importance_top20.png` - Bar chart of top 20 features
- `feature_importance_report.json` - Detailed SHAP values

---

### 2. **Per-Transaction Explanations**

Get detailed explanation for individual transactions including:
- Top contributing features with SHAP values
- Natural language reasoning
- Confidence breakdown by component

**Usage:**
```bash
# Explain single transaction
python explainability.py --mode single --merchant "STARBUCKS #12345" --amount 4.75
```

**Output:**
```
Transaction: STARBUCKS #12345 ($4.75)

Prediction:
  Category: Dining - Coffee Shops - Starbucks
  Confidence: 95.2%

Transaction 'STARBUCKS #12345' ($4.75) was categorized as
'Dining - Coffee Shops - Starbucks' with 95.2% confidence.

This is a high-confidence prediction.

Key factors in this decision:
  • Spending amount range (micro) increased confidence in this category
  • Transaction timing pattern (daily) increased confidence
  • Merchant name semantic similarity to known 'Dining - Coffee Shops - Starbucks' merchants

Category hierarchy: Dining → Dining - Coffee Shops → Dining - Coffee Shops - Starbucks
```

**Saved to:** `explainability_results/explanation.json`

---

### 3. **Confidence Breakdown**

Detailed breakdown of confidence score components:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Model Probability** | 70% | LightGBM confidence based on learned patterns |
| **MCC Code Match** | 20% | Merchant Category Code correlation |
| **Hierarchical Consistency** | 10% | Category hierarchy validation (L1 → L2 → L3) |

**Example:**
```json
{
  "overall_confidence": 0.952,
  "components": [
    {
      "component": "model_probability",
      "weight": 0.7,
      "description": "LightGBM model confidence based on learned patterns"
    },
    {
      "component": "mcc_code_match",
      "weight": 0.2,
      "description": "MCC code 5814 correlation with category"
    },
    {
      "component": "hierarchical_consistency",
      "weight": 0.1,
      "description": "Category hierarchy validation (L1 → L2 → L3)"
    }
  ]
}
```

---

### 4. **Top Contributing Features**

For each prediction, get the top 10 features that influenced the decision:

**Example:**
```json
{
  "top_features": [
    {
      "feature": "spend_band",
      "value": 0.0,
      "shap_value": 0.342,
      "impact": "positive"
    },
    {
      "feature": "embedding_dim_45",
      "value": 0.523,
      "shap_value": 0.198,
      "impact": "positive"
    },
    {
      "feature": "temporal_pattern",
      "value": 0.0,
      "shap_value": 0.156,
      "impact": "positive"
    }
  ]
}
```

**Interpretation:**
- **Positive impact**: Feature increased likelihood of predicted category
- **Negative impact**: Feature decreased likelihood of predicted category
- **SHAP value**: Magnitude of impact

---

### 5. **Natural Language Reasoning**

Human-readable explanation of why the prediction was made:

**Example:**
```
Transaction 'AMAZON.COM' ($29.99) was categorized as
'Shopping - Online - Amazon' with 87.3% confidence.

This is a high-confidence prediction.

Key factors in this decision:
  • Spending amount range (low) increased confidence in this category
  • Transaction channel (online) increased likelihood of this category
  • Merchant Category Code increased confidence
  • Merchant name semantic similarity to known 'Shopping - Online - Amazon' merchants

Category hierarchy: Shopping → Shopping - Online → Shopping - Online - Amazon
```

---

## 🔍 Feature Types Explained

### **Semantic Embedding Features (768 dimensions)**
- Capture meaning and context of merchant names
- Learned representations from Sentence-BERT model
- Example: `embedding_dim_0` through `embedding_dim_767`

**Use case:** Identifying similar merchants by semantic meaning (e.g., "Starbucks" and "SBUX" both map to coffee shops)

### **Engineered Features (5 dimensions)**
- **spend_band**: Categorizes amount into tiers (micro, low, medium, high, premium)
- **temporal_pattern**: Transaction timing (daily, weekly, monthly, irregular)
- **channel**: Transaction method (online, pos, atm, mobile)
- **mcc_code_normalized**: Merchant Category Code (0-1 scale)
- **amount_percentile**: Amount relative to all transactions (0-1 scale)

**Use case:** Capturing domain-specific patterns (e.g., coffee shops typically have small daily transactions)

---

## 📊 SHAP Analysis

### What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain machine learning predictions:
- **Fair attribution**: Each feature gets credit proportional to its contribution
- **Model-agnostic**: Works with any ML model
- **Additive**: SHAP values sum to the difference between prediction and base rate

### SHAP Value Interpretation

```
Prediction = Base Value + SHAP(feature1) + SHAP(feature2) + ... + SHAP(feature773)
```

- **Base Value**: Average prediction across all training data
- **Positive SHAP**: Feature pushed prediction higher
- **Negative SHAP**: Feature pushed prediction lower
- **Zero SHAP**: Feature had no impact

### Example SHAP Visualization

```
Feature                      SHAP Value      Impact
========================================================
spend_band                    +0.342         ████████████░░ Positive
embedding_dim_45              +0.198         ███████░░░░░░░ Positive
temporal_pattern              +0.156         ██████░░░░░░░░ Positive
channel                       +0.089         ███░░░░░░░░░░░ Positive
mcc_code_normalized           -0.023         █░░░░░░░░░░░░░ Negative
```

---

## 🚀 Usage Examples

### Example 1: Explain a Coffee Shop Transaction

```bash
python explainability.py \
  --mode single \
  --merchant "STARBUCKS STORE #4532" \
  --amount 5.25
```

**Expected Output:**
- Category: Dining - Coffee Shops - Starbucks
- Confidence: 90-95%
- Key factors: spend_band (micro), semantic similarity, temporal_pattern (daily)

---

### Example 2: Explain a High-Value Transaction

```bash
python explainability.py \
  --mode single \
  --merchant "APPLE STORE ONLINE" \
  --amount 1299.00
```

**Expected Output:**
- Category: Shopping - Retail - Electronics or Technology - Electronics - Computer
- Confidence: 75-85% (may be ambiguous)
- Key factors: spend_band (premium), channel (online), merchant semantics

---

### Example 3: Generate Feature Importance Report

```bash
python explainability.py \
  --mode batch \
  --data data/test.csv \
  --output explainability_results
```

**Generates:**
- `feature_importance_top20.png` - Visual bar chart
- `feature_importance_report.json` - Detailed rankings

**Use case:** Understand which features are most important globally across all predictions

---

## 📈 Expected Results

Based on the model architecture (768D embeddings + 5 engineered features = 773 total):

### Typical Top 20 Features:
1. **Semantic embeddings** (60-70%): Capture merchant name patterns
2. **spend_band** (Top 5): Amount tier is highly predictive
3. **temporal_pattern** (Top 10): Transaction frequency matters
4. **channel** (Top 15): Online vs POS vs ATM
5. **mcc_code_normalized** (Top 20): MCC codes add signal

### Feature Category Distribution:
- Semantic Embeddings: 12-14 of top 20
- Engineered Features: 6-8 of top 20

**Insight:** Both semantic meaning AND domain features are important!

---

## ⚠️ Limitations & Considerations

### 1. **SHAP Computation Time**
- SHAP analysis is computationally expensive
- Recommended: Use subset of data (100-500 samples) for global analysis
- Single transaction explanations are fast (~1-2 seconds)

### 2. **Interpretability vs Accuracy Trade-off**
- Engineered features are more interpretable
- Semantic embeddings are more accurate but harder to interpret
- Both are needed for best results

### 3. **Confidence ≠ Correctness**
- High confidence doesn't guarantee correctness
- Low-confidence predictions should be manually reviewed
- Use confidence thresholds (e.g., 70% or 80%) for production

### 4. **Context Dependencies**
- Explanations are specific to the trained model
- Retraining may change feature importance
- Always validate explanations match business logic

---

## 🎯 Best Practices

### For Production Deployment:

1. **Threshold-based Review:**
   ```python
   if confidence < 0.70:
       # Send to manual review queue
       # Include explanation to help reviewer
   ```

2. **Feature Importance Monitoring:**
   - Regenerate feature importance monthly
   - Alert if top features change significantly
   - May indicate data drift

3. **Explanation Logging:**
   ```python
   # Log explanations for audit trail
   {
       "transaction_id": "txn_12345",
       "prediction": "Dining - Coffee - Starbucks",
       "confidence": 0.952,
       "top_features": [...],
       "reasoning": "..."
   }
   ```

4. **User-Facing Explanations:**
   - Show confidence score
   - Show category hierarchy
   - Show 3-5 key factors (in plain language)
   - Don't show SHAP values to end users

---

## 📝 Integration Examples

### Python API Integration:

```python
from explainability import ExplainabilityEngine

# Initialize
engine = ExplainabilityEngine(model_path="models")

# Get explanation
transaction = {
    "merchant": "WHOLE FOODS MARKET",
    "amount": 87.50,
    "currency": "USD",
    "timestamp": "2024-11-22T10:30:00",
    "channel": "pos"
}

explanation = engine.explain_prediction(transaction, top_k_features=5)

# Use explanation
print(f"Category: {explanation['prediction']['l3']}")
print(f"Confidence: {explanation['prediction']['confidence']*100:.1f}%")
print(f"Reasoning: {explanation['reasoning']}")
```

### Web API Integration:

```python
from fastapi import FastAPI
from explainability import ExplainabilityEngine

app = FastAPI()
engine = ExplainabilityEngine()

@app.post("/categorize_with_explanation")
def categorize(transaction: dict):
    explanation = engine.explain_prediction(transaction)
    return {
        "category": explanation['prediction']['l3'],
        "confidence": explanation['prediction']['confidence'],
        "reasoning": explanation['reasoning'],
        "top_features": explanation['top_features'][:3]  # Top 3 only for API
    }
```

---

## 🔬 Technical Details

### SHAP Explainer Configuration:

```python
shap_explainer = shap.TreeExplainer(
    model=lightgbm_model,
    data=background_data,  # 100 samples for efficiency
    feature_perturbation="tree_path_dependent"  # Fast approximation
)
```

**Parameters:**
- `background_data`: Reference distribution for SHAP computation
- `feature_perturbation="tree_path_dependent"`: Fast approximation (vs exact)
- Uses TreeExplainer optimized for LightGBM

### Computational Complexity:

| Operation | Complexity | Time (Single Txn) | Time (100 Txns) |
|-----------|------------|-------------------|-----------------|
| Prediction | O(depth × trees) | ~10ms | ~1s |
| SHAP values | O(features × trees) | ~100ms | ~10s |
| Feature importance | O(samples × features × trees) | N/A | ~30s |

**Recommendation:** Compute SHAP values on-demand, not for every prediction

---

## 📚 Further Reading

- [SHAP Documentation](https://shap.readthedocs.io/)
- [TreeExplainer for LightGBM](https://shap.readthedocs.io/en/latest/example_notebooks/tree_based_models/Tree%20SHAP%20Algorithms.html)
- [Interpreting Machine Learning Models](https://christophm.github.io/interpretable-ml-book/)

---

## ✅ Summary

Holmes AI now provides:

✅ **SHAP-based feature importance** - Know which features matter most
✅ **Per-transaction explanations** - Understand individual predictions
✅ **Natural language reasoning** - Human-readable explanations
✅ **Confidence breakdown** - Transparency in confidence scores
✅ **Top contributing features** - See what drove each decision

**Explainability Status:** **80% Complete** ✅

- ✅ SHAP feature importance
- ✅ Confidence breakdown
- ✅ Natural language reasoning
- ✅ Top features analysis
- ⚠️ Interactive UI (optional enhancement)

---

**Generated:** 2024-11-22
**Module:** explainability.py
**Dependencies:** shap, numpy, pandas, matplotlib
