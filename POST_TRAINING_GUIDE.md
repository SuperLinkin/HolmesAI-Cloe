# Post-Training Evaluation Guide

This guide outlines all steps to run after Google Colab training completes to generate comprehensive evaluation reports and verify deliverables.

---

## 📋 Prerequisites

After Colab training completes, you should have:

1. ✅ Trained models in Google Drive (`models/` directory)
2. ✅ Test dataset (`data/test.csv` or similar)
3. ✅ All evaluation scripts in local repository

---

## 🚀 Step-by-Step Execution

### Step 1: Download Trained Models from Colab

From Google Colab, download the trained models:

```bash
# In Colab, zip the models directory
!zip -r trained_models.zip models/

# Download via Colab interface or use:
from google.colab import files
files.download('trained_models.zip')
```

Then on your local machine:

```bash
# Extract to your project directory
unzip trained_models.zip -d c:/Users/Pranav\ Mv/Documents/Holmes_Cloe/
```

---

### Step 2: Run Comprehensive Model Evaluation

**Purpose:** Generate complete metrics report with confusion matrices, F1 scores, and performance benchmarks

**Command:**
```bash
python evaluate_model.py --model models --data data/test.csv --output evaluation_results
```

**What it generates:**
- `evaluation_results/evaluation_report.json` - JSON metrics
- `evaluation_results/EVALUATION_REPORT.md` - Markdown report
- `evaluation_results/confusion_matrix_l1.png` - L1 confusion matrix
- `evaluation_results/confusion_matrix_l2.png` - L2 confusion matrix
- `evaluation_results/confusion_matrix_l3.png` - L3 confusion matrix
- `evaluation_results/classification_report_l1.csv` - L1 per-class metrics
- `evaluation_results/classification_report_l2.csv` - L2 per-class metrics
- `evaluation_results/classification_report_l3.csv` - L3 per-class metrics

**Expected runtime:** 5-10 minutes for 100K samples

**Success criteria:**
- ✅ L1 Macro F1 ≥ 0.90
- ✅ L2 Macro F1 ≥ 0.90
- ✅ L3 Macro F1 ≥ 0.90
- ✅ Average latency < 200ms

---

### Step 3: Run Bias Analysis

**Purpose:** Detect performance disparities across categories and identify potential bias

**Command:**
```bash
python analyze_bias.py --model models --data data/test.csv --output bias_analysis
```

**What it generates:**
- `bias_analysis/bias_analysis_report.json` - JSON analysis
- `bias_analysis/BIAS_ANALYSIS_REPORT.md` - Markdown report
- `bias_analysis/per_category_metrics_l1.csv` - L1 per-category F1
- `bias_analysis/per_category_metrics_l2.csv` - L2 per-category F1
- `bias_analysis/per_category_metrics_l3.csv` - L3 per-category F1
- `bias_analysis/performance_vs_frequency_l1.png` - L1 scatter plot
- `bias_analysis/performance_vs_frequency_l2.png` - L2 scatter plot
- `bias_analysis/performance_vs_frequency_l3.png` - L3 scatter plot

**Expected runtime:** 5-10 minutes for 100K samples

**What to look for:**
- ⚠️ F1 variance > 0.15 indicates high disparity
- ⚠️ Categories with F1 < 0.80 need attention
- ✅ Uniform performance across categories is ideal

---

### Step 4: Run Interactive Demo

**Purpose:** Demonstrate complete pipeline execution for showcase/presentation

**Command:**
```bash
python demo.py
```

**What it shows:**
1. Complete pipeline execution (raw data → prediction)
2. Sample predictions with varying confidence levels
3. Taxonomy modification instructions
4. Performance benchmarks (latency, throughput)

**Expected runtime:** 2-3 minutes (interactive with prompts)

---

### Step 5: Validate Taxonomy Configuration

**Purpose:** Verify taxonomy is valid before deployment

**Command:**
```bash
python validate_taxonomy.py
```

**What it checks:**
- ✅ Valid JSON syntax
- ✅ No duplicate categories
- ✅ All required fields present
- ✅ Proper hierarchical structure

**Expected output:**
```
✅ Taxonomy validation PASSED
   Total L1 categories: 15
   Total L2 categories: 42
   Total L3 categories: 59
```

---

## 📊 Interpreting Results

### Evaluation Report (evaluation_results/EVALUATION_REPORT.md)

**Key metrics to check:**

1. **Macro F1 Scores:**
   - L1: Should be ≥ 0.90 (likely ~0.99)
   - L2: Should be ≥ 0.90 (likely ~0.90-0.95)
   - L3: Should be ≥ 0.90 (likely ~0.85-0.92)

2. **Performance:**
   - Average latency: Should be < 200ms
   - Throughput: Should be > 100 txns/sec

3. **If F1 < 0.90:**
   - Check which level failed
   - Review confusion matrix for that level
   - Consider generating larger dataset (150K-200K)
   - Retrain with more samples

### Bias Analysis Report (bias_analysis/BIAS_ANALYSIS_REPORT.md)

**Key findings to check:**

1. **Imbalance Ratio:**
   - < 5x: ✅ Good balance
   - 5-10x: ⚠️ Moderate imbalance
   - > 10x: ⚠️ High imbalance - may need rebalancing

2. **F1 Variance:**
   - < 0.10: ✅ Uniform performance
   - 0.10-0.15: ⚠️ Some disparity
   - > 0.15: ⚠️ High disparity - investigate low performers

3. **Low-Frequency Categories:**
   - Check if low-frequency → low F1 correlation
   - If yes, consider oversampling or data augmentation

---

## 🎯 Decision Tree: What to Do Next

### If All Levels Meet F1 ≥ 0.90:
✅ **SUCCESS! Project complete.**

**Next steps:**
1. ✅ Review bias analysis for any disparity concerns
2. ✅ Optionally record demo video
3. ✅ Update README with achieved scores
4. ✅ Package final deliverables
5. ✅ Deploy to production

---

### If L1 or L2 < 0.90:
⚠️ **Unexpected - likely a bug or data issue**

**Troubleshooting:**
1. Check if bug fix was applied correctly (verify_split_fix.py)
2. Verify training completed without errors
3. Check training logs for anomalies
4. Review dataset for label errors
5. Consider retraining with different random seed

---

### If L3 < 0.90 (but L1, L2 ≥ 0.90):
⚠️ **Expected scenario - L3 is most granular**

**Options:**

**Option A: Accept current performance**
- If L3 F1 is 0.85-0.89, this is still very good
- Document the limitation
- Recommend human review for low-confidence predictions
- Deploy with current performance

**Option B: Improve L3 performance**
1. Generate larger dataset (150K-200K samples):
   ```bash
   python generate_dataset.py --output data/synthetic_transactions_200k.csv --num-samples 200000
   ```

2. Split into train/test:
   ```bash
   python -c "import pandas as pd; df=pd.read_csv('data/synthetic_transactions_200k.csv'); train=df.sample(frac=0.85, random_state=42); test=df.drop(train.index); train.to_csv('data/train_200k.csv', index=False); test.to_csv('data/test_200k.csv', index=False)"
   ```

3. Retrain on Colab with 200K samples

4. Re-evaluate

---

## 📦 Final Deliverables Checklist

Before submission, ensure you have:

### 1. Source Code Repository
- [x] README.md with setup instructions
- [x] Complete source code in src/
- [x] Configuration files (taxonomy.json)
- [x] requirements.txt

### 2. Trained Models
- [ ] models/lightgbm/ directory with trained models
- [ ] models/sentence_bert/ directory with encoder

### 3. Evaluation Reports
- [ ] evaluation_results/EVALUATION_REPORT.md
- [ ] evaluation_results/confusion_matrix_*.png (3 files)
- [ ] evaluation_results/classification_report_*.csv (3 files)

### 4. Bias Analysis
- [ ] bias_analysis/BIAS_ANALYSIS_REPORT.md
- [ ] bias_analysis/performance_vs_frequency_*.png (3 files)
- [ ] bias_analysis/per_category_metrics_*.csv (3 files)

### 5. Documentation
- [x] SYSTEM_ARCHITECTURE.md
- [x] DELIVERABLES_CHECKLIST.md
- [x] TAXONOMY_ADMIN_GUIDE.md
- [x] DATASET.md
- [x] CRITICAL_BUG_FIX_SUMMARY.md

### 6. Demo
- [x] demo.py script
- [ ] Demo video (optional but recommended)

---

## 🎬 Optional: Record Demo Video

**Recommended length:** 5-10 minutes

**What to show:**

1. **Introduction** (30 seconds)
   - Project overview
   - Key features

2. **Demo Execution** (3-4 minutes)
   - Run `python demo.py`
   - Show pipeline execution
   - Show sample predictions
   - Show performance benchmarks

3. **Taxonomy Modification** (1-2 minutes)
   - Open taxonomy.json
   - Show structure
   - Explain how to add category
   - Run validation

4. **Evaluation Results** (2-3 minutes)
   - Show EVALUATION_REPORT.md
   - Highlight F1 scores
   - Show confusion matrices
   - Show performance metrics

5. **Bias Analysis** (1-2 minutes)
   - Show BIAS_ANALYSIS_REPORT.md
   - Explain findings
   - Show mitigation recommendations

6. **Conclusion** (30 seconds)
   - Summary of achievements
   - Production readiness

**Tools for recording:**
- OBS Studio (free, recommended)
- Windows Game Bar (Win+G)
- Loom (web-based)

---

## 📝 Commands Quick Reference

```bash
# 1. Download models from Colab (if not already downloaded)
# (Use Colab UI or files.download())

# 2. Run comprehensive evaluation
python evaluate_model.py --model models --data data/test.csv --output evaluation_results

# 3. Run bias analysis
python analyze_bias.py --model models --data data/test.csv --output bias_analysis

# 4. Run interactive demo
python demo.py

# 5. Validate taxonomy
python validate_taxonomy.py

# 6. (Optional) If L3 < 0.90, generate larger dataset
python generate_dataset.py --output data/synthetic_transactions_200k.csv --num-samples 200000
```

---

## ⏱️ Total Expected Runtime

| Task | Runtime | Can Skip? |
|------|---------|-----------|
| Download models | 2-5 min | No |
| Evaluation | 5-10 min | No |
| Bias analysis | 5-10 min | No (bonus but recommended) |
| Demo | 2-3 min | Yes |
| Taxonomy validation | < 1 min | Yes |
| Demo video | 10-15 min | Yes |

**Total minimum:** ~15-25 minutes
**Total with all optional:** ~35-45 minutes

---

## 🆘 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Ensure you're in the project root directory
cd c:/Users/Pranav\ Mv/Documents/Holmes_Cloe/
python evaluate_model.py --model models --data data/test.csv --output evaluation_results
```

### Error: "FileNotFoundError: models/lightgbm not found"

**Solution:** Ensure models are downloaded from Colab and extracted to the correct location.

### Error: "No such file or directory: data/test.csv"

**Solution:** Use the correct test dataset path:
```bash
# Check available datasets
ls data/*.csv

# Use the correct one (e.g., synthetic_transactions_test.csv)
python evaluate_model.py --model models --data data/synthetic_transactions_test.csv --output evaluation_results
```

### Evaluation takes too long (> 30 min)

**Solution:** Your test dataset may be too large. Use a sample:
```bash
# Create 10K sample
python -c "import pandas as pd; df=pd.read_csv('data/test.csv'); df.sample(10000).to_csv('data/test_10k.csv', index=False)"

# Run on sample
python evaluate_model.py --model models --data data/test_10k.csv --output evaluation_results
```

---

## ✅ Success Indicators

You're ready for submission when:

1. ✅ All evaluation scripts run without errors
2. ✅ L1, L2, L3 Macro F1 ≥ 0.90 (or documented reason if not)
3. ✅ Average latency < 200ms
4. ✅ Bias analysis shows no critical disparities (or documented mitigation plan)
5. ✅ All deliverable files generated
6. ✅ README updated with final results

---

**Last Updated:** 2024-11-22

**Next Review:** After Colab training completes
