# Holmes AI - Bias Analysis Report

**Generated:** 2025-11-22T17:11:33.596500

**Model Path:** models

**Test Dataset:** data/test.csv

**Test Samples:** 10,000

---

## Executive Summary

This report analyzes model performance across all categories to detect potential bias and performance disparities.

### Key Findings


#### L1

- **Categories:** 15
- **F1 Score Range:** 0.9536 - 1.0000
- **Mean F1:** 0.9960 ± 0.0116
- **Categories Below 0.90:** 0/15 (0.0%)
- **Categories Below 0.80:** 0/15
- ✅ Variance within acceptable range

#### L2

- **Categories:** 42
- **F1 Score Range:** 0.7927 - 1.0000
- **Mean F1:** 0.9792 ± 0.0505
- **Categories Below 0.90:** 4/42 (9.5%)
- **Categories Below 0.80:** 1/42
- ✅ Variance within acceptable range

#### L3

- **Categories:** 59
- **F1 Score Range:** 0.7927 - 1.0000
- **Mean F1:** 0.9728 ± 0.0557
- **Categories Below 0.90:** 11/59 (18.6%)
- **Categories Below 0.80:** 1/59
- ✅ Variance within acceptable range

---

## Category Distribution

### Imbalance Analysis


#### L1

| Metric | Value |
|--------|-------|
| Total Categories | 15 |
| Min Samples | 86 |
| Max Samples | 2170 |
| Mean Samples | 666.7 |
| Std Dev | 569.8 |
| **Imbalance Ratio** | **25.23x** |

⚠️ **HIGH IMBALANCE** - Significant disparity between most/least frequent categories


#### L2

| Metric | Value |
|--------|-------|
| Total Categories | 42 |
| Min Samples | 86 |
| Max Samples | 555 |
| Mean Samples | 238.1 |
| Std Dev | 175.7 |
| **Imbalance Ratio** | **6.45x** |

⚠️ **MODERATE IMBALANCE** - Some categories underrepresented


#### L3

| Metric | Value |
|--------|-------|
| Total Categories | 59 |
| Min Samples | 86 |
| Max Samples | 284 |
| Mean Samples | 169.5 |
| Std Dev | 74.6 |
| **Imbalance Ratio** | **3.30x** |

✅ Relatively balanced distribution


---

## Performance vs Frequency Analysis

Plots showing relationship between category frequency and F1 score:

- [L1 Performance vs Frequency](performance_vs_frequency_l1.png)
- [L2 Performance vs Frequency](performance_vs_frequency_l2.png)
- [L3 Performance vs Frequency](performance_vs_frequency_l3.png)

**Interpretation:** If low-frequency categories consistently have lower F1 scores, this indicates sample bias.

---

## Low-Performing Categories


### L1 Categories Below 0.80 F1

✅ All categories meet minimum threshold (F1 ≥ 0.80)


### L2 Categories Below 0.80 F1

⚠️ 1 categories underperforming:

| Category | F1 Score | Samples | Precision | Recall |
|----------|----------|---------|-----------|--------|
| Charitable - Donations | 0.7927 | 92 | 0.9028 | 0.7065 |


### L3 Categories Below 0.80 F1

⚠️ 1 categories underperforming:

| Category | F1 Score | Samples | Precision | Recall |
|----------|----------|---------|-----------|--------|
| Charitable - Donations - NGO | 0.7927 | 92 | 0.9028 | 0.7065 |


---

## Mitigation Recommendations

1. **L1 Class Imbalance:** Implement oversampling/SMOTE for low-frequency categories (bottom 10%)
2. **L2 Low Performers:** Increase training samples for 1 underperforming categories to at least 2-3x current count
3. **L3 Low Performers:** Increase training samples for 1 underperforming categories to at least 2-3x current count


### General Mitigation Strategies

1. **Data Augmentation:** Generate synthetic samples for low-frequency categories using techniques like back-translation or paraphrasing
2. **Class Weighting:** Already implemented - may need tuning for specific underperforming categories
3. **Focal Loss:** Implement focal loss to focus training on hard-to-classify examples
4. **Ensemble Methods:** Train separate models for low-frequency categories
5. **Active Learning:** Prioritize labeling of misclassified low-frequency examples
6. **Hierarchical Balancing:** Balance samples at each hierarchy level independently

---

## Detailed Metrics

Per-category metrics are available in:

- [L1 Per-Category Metrics](per_category_metrics_l1.csv)
- [L2 Per-Category Metrics](per_category_metrics_l2.csv)
- [L3 Per-Category Metrics](per_category_metrics_l3.csv)

---

## Files Generated

- `bias_analysis_report.json` - Complete analysis in JSON format
- `per_category_metrics_l1.csv` - L1 per-category metrics
- `per_category_metrics_l2.csv` - L2 per-category metrics
- `per_category_metrics_l3.csv` - L3 per-category metrics
- `performance_vs_frequency_l1.png` - L1 performance vs sample count plot
- `performance_vs_frequency_l2.png` - L2 performance vs sample count plot
- `performance_vs_frequency_l3.png` - L3 performance vs sample count plot
- `BIAS_ANALYSIS_REPORT.md` - This report

---

**Note:** This analysis should be performed periodically as new data is collected to ensure the model remains fair across all categories.
