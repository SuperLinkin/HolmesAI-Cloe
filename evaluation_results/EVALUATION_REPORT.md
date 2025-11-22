# Holmes AI - Model Evaluation Report

**Generated:** 2025-11-22T16:55:03.658146

**Model Path:** models

**Test Dataset:** data/test.csv

**Test Samples:** 10,000

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Average Latency | 10.22ms |
| P95 Latency | 11.27ms |
| P99 Latency | 12.06ms |
| Throughput | 486 txns/sec |
| Embedding Time | 205.61s |

---

## Accuracy Metrics

### Summary

| Level | Accuracy | Macro F1 | Weighted F1 | Classes | Target (≥0.90) |
|-------|----------|----------|-------------|---------|----------------|
| **L1** | 0.9964 (99.64%) | **0.9960** | 0.9964 | 15 | ✅ |
| **L2** | 0.9848 (98.48%) | **0.9792** | 0.9847 | 42 | ✅ |
| **L3** | 0.9753 (97.53%) | **0.9728** | 0.9752 | 59 | ✅ |

### Target Achievement

**Overall Target (All levels Macro F1 ≥ 0.90):** ✅ **ACHIEVED**

---

## Confusion Matrices

Confusion matrices have been generated for each level:

- [L1 Confusion Matrix](confusion_matrix_l1.png)
- [L2 Confusion Matrix](confusion_matrix_l2.png)
- [L3 Confusion Matrix](confusion_matrix_l3.png)

---

## Detailed Classification Reports

Per-class metrics are available in:

- [L1 Classification Report](classification_report_l1.csv)
- [L2 Classification Report](classification_report_l2.csv)
- [L3 Classification Report](classification_report_l3.csv)

---

## Recommendations


✅ **All levels meet the target Macro F1 ≥ 0.90**

The model is production-ready with excellent performance across all hierarchical levels.


---

## Files Generated

- `evaluation_report.json` - Complete evaluation metrics in JSON format
- `confusion_matrix_l1.png` - L1 confusion matrix visualization
- `confusion_matrix_l2.png` - L2 confusion matrix visualization
- `confusion_matrix_l3.png` - L3 confusion matrix visualization
- `classification_report_l1.csv` - L1 per-class metrics
- `classification_report_l2.csv` - L2 per-class metrics
- `classification_report_l3.csv` - L3 per-class metrics
- `EVALUATION_REPORT.md` - This report
