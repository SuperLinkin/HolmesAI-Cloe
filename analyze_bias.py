"""
Bias Analysis and Mitigation Tool

Analyzes model performance across categories to detect bias:
- Per-category F1 scores
- Low-frequency category detection
- Performance disparity analysis
- Recommendations for mitigation
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_ingestion import DataIngestion
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier


def analyze_category_distribution(transactions):
    """Analyze distribution of categories in dataset."""

    l1_counts = Counter([t['l1'] for t in transactions])
    l2_counts = Counter([t['l2'] for t in transactions])
    l3_counts = Counter([t['l3'] for t in transactions])

    return {
        'L1': {
            'counts': dict(l1_counts),
            'num_categories': len(l1_counts),
            'min_samples': min(l1_counts.values()),
            'max_samples': max(l1_counts.values()),
            'mean_samples': np.mean(list(l1_counts.values())),
            'std_samples': np.std(list(l1_counts.values()))
        },
        'L2': {
            'counts': dict(l2_counts),
            'num_categories': len(l2_counts),
            'min_samples': min(l2_counts.values()),
            'max_samples': max(l2_counts.values()),
            'mean_samples': np.mean(list(l2_counts.values())),
            'std_samples': np.std(list(l2_counts.values()))
        },
        'L3': {
            'counts': dict(l3_counts),
            'num_categories': len(l3_counts),
            'min_samples': min(l3_counts.values()),
            'max_samples': max(l3_counts.values()),
            'mean_samples': np.mean(list(l3_counts.values())),
            'std_samples': np.std(list(l3_counts.values()))
        }
    }


def calculate_per_category_metrics(y_true, y_pred, level_name):
    """Calculate metrics for each category."""

    categories = sorted(list(set(y_true) | set(y_pred)))

    metrics = []
    for category in categories:
        # Binary classification for this category
        y_true_binary = [1 if y == category else 0 for y in y_true]
        y_pred_binary = [1 if y == category else 0 for y in y_pred]

        # Calculate metrics
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # Count samples
        num_samples = sum(y_true_binary)
        num_correct = sum(1 for t, p in zip(y_true, y_pred) if t == category and p == category)

        metrics.append({
            'category': category,
            'num_samples': num_samples,
            'num_correct': num_correct,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    return pd.DataFrame(metrics)


def identify_low_performance_categories(metrics_df, threshold=0.80):
    """Identify categories with F1 score below threshold."""

    low_performers = metrics_df[metrics_df['f1_score'] < threshold].sort_values('f1_score')
    return low_performers


def identify_low_frequency_categories(distribution, percentile=10):
    """Identify categories in bottom N percentile by sample count."""

    counts = list(distribution['counts'].values())
    threshold = np.percentile(counts, percentile)

    low_freq = {
        cat: count for cat, count in distribution['counts'].items()
        if count <= threshold
    }

    return low_freq, threshold


def plot_performance_vs_frequency(metrics_df, distribution, level_name, save_path):
    """Plot F1 score vs number of samples to detect bias."""

    # Merge metrics with sample counts
    metrics_df['sample_count'] = metrics_df['category'].map(distribution['counts'])

    plt.figure(figsize=(12, 6))

    # Scatter plot
    plt.scatter(metrics_df['sample_count'], metrics_df['f1_score'], alpha=0.6)

    # Add trend line
    z = np.polyfit(metrics_df['sample_count'], metrics_df['f1_score'], 1)
    p = np.poly1d(z)
    plt.plot(metrics_df['sample_count'], p(metrics_df['sample_count']),
             "r--", alpha=0.8, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2f}')

    # Reference lines
    plt.axhline(y=0.90, color='g', linestyle='--', alpha=0.5, label='Target F1=0.90')
    plt.axhline(y=0.80, color='orange', linestyle='--', alpha=0.5, label='Warning F1=0.80')

    plt.xlabel('Number of Training Samples')
    plt.ylabel('F1 Score')
    plt.title(f'{level_name} Performance vs Sample Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {save_path}")


def analyze_bias(
    model_path: str,
    data_path: str,
    output_dir: str = "bias_analysis"
):
    """
    Analyze model bias and performance disparities.

    Args:
        model_path: Path to saved models directory
        data_path: Path to test dataset
        output_dir: Directory to save analysis results
    """

    print("=" * 80)
    print("HOLMES AI - BIAS ANALYSIS")
    print("=" * 80)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Step 1: Load test data
    print("[1/6] Loading test data...")
    ingestion = DataIngestion()
    df = pd.read_csv(data_path)
    normalized = ingestion.ingest_pipeline(data_path)

    transactions = []
    for i, txn in enumerate(normalized):
        txn_dict = txn.model_dump()
        row = df.iloc[i]
        txn_dict['l1'] = row['l1']
        txn_dict['l2'] = row['l2']
        txn_dict['l3'] = row['l3']
        transactions.append(txn_dict)

    print(f"[OK] Loaded {len(transactions):,} test transactions")

    # Step 2: Analyze category distribution
    print("\n[2/6] Analyzing category distribution...")
    distribution = analyze_category_distribution(transactions)

    for level in ['L1', 'L2', 'L3']:
        dist = distribution[level]
        print(f"\n{level} Distribution:")
        print(f"  Categories: {dist['num_categories']}")
        print(f"  Min samples: {dist['min_samples']}")
        print(f"  Max samples: {dist['max_samples']}")
        print(f"  Mean samples: {dist['mean_samples']:.1f}")
        print(f"  Std samples: {dist['std_samples']:.1f}")
        print(f"  Imbalance ratio: {dist['max_samples'] / dist['min_samples']:.2f}x")

    # Step 3: Preprocess and load models
    print("\n[3/6] Loading models...")
    preprocessor = TransactionPreprocessor()
    preprocessed = preprocessor.preprocess_batch(transactions)

    encoder = SentenceBERTEncoder(model_path=f"{model_path}/sentence_bert")
    classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")
    classifier.load_models(f"{model_path}/lightgbm")
    print("[OK] Models loaded")

    # Step 4: Generate predictions
    print("\n[4/6] Generating predictions...")
    embeddings = encoder.encode_transactions(
        preprocessed,
        text_field='merchant_cleaned',
        batch_size=64
    )
    X = classifier.prepare_features(embeddings, preprocessed, include_enrichment=True)
    predictions = classifier.predict(X, use_hierarchy=True)

    y_true_l1 = [t['l1'] for t in preprocessed]
    y_true_l2 = [t['l2'] for t in preprocessed]
    y_true_l3 = [t['l3'] for t in preprocessed]

    y_pred_l1 = predictions['l1']
    y_pred_l2 = predictions['l2']
    y_pred_l3 = predictions['l3']

    print("[OK] Predictions generated")

    # Step 5: Calculate per-category metrics
    print("\n[5/6] Calculating per-category metrics...")

    results = {}

    for level, y_true, y_pred, dist in [
        ('L1', y_true_l1, y_pred_l1, distribution['L1']),
        ('L2', y_true_l2, y_pred_l2, distribution['L2']),
        ('L3', y_true_l3, y_pred_l3, distribution['L3'])
    ]:
        print(f"\nAnalyzing {level}...")

        # Per-category metrics
        metrics_df = calculate_per_category_metrics(y_true, y_pred, level)

        # Save detailed metrics
        metrics_df.to_csv(output_path / f'per_category_metrics_{level.lower()}.csv', index=False)

        # Identify low performers
        low_performers = identify_low_performance_categories(metrics_df, threshold=0.80)

        # Identify low frequency categories
        low_freq, freq_threshold = identify_low_frequency_categories(dist, percentile=10)

        # Plot performance vs frequency
        plot_performance_vs_frequency(
            metrics_df.copy(),
            dist,
            level,
            output_path / f'performance_vs_frequency_{level.lower()}.png'
        )

        # Calculate statistics
        f1_scores = metrics_df['f1_score'].values

        results[level] = {
            'num_categories': len(metrics_df),
            'mean_f1': float(np.mean(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
            'min_f1': float(np.min(f1_scores)),
            'max_f1': float(np.max(f1_scores)),
            'categories_below_0.80': int(len(low_performers)),
            'categories_below_0.90': int(len(metrics_df[metrics_df['f1_score'] < 0.90])),
            'low_frequency_threshold': int(freq_threshold),
            'num_low_frequency': len(low_freq),
            'low_performers': low_performers.to_dict('records'),
            'low_frequency_categories': low_freq
        }

        print(f"  Mean F1: {results[level]['mean_f1']:.4f}")
        print(f"  Std F1: {results[level]['std_f1']:.4f}")
        print(f"  Min F1: {results[level]['min_f1']:.4f}")
        print(f"  Max F1: {results[level]['max_f1']:.4f}")
        print(f"  Categories < 0.80 F1: {results[level]['categories_below_0.80']}")
        print(f"  Categories < 0.90 F1: {results[level]['categories_below_0.90']}")
        print(f"  Low frequency categories (bottom 10%): {results[level]['num_low_frequency']}")

    # Step 6: Generate bias report
    print("\n[6/6] Generating bias analysis report...")

    report = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'model_path': model_path,
        'test_data': data_path,
        'test_samples': len(transactions),
        'distribution': distribution,
        'bias_analysis': results
    }

    # Save JSON report
    with open(output_path / 'bias_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Generate markdown report
    generate_bias_report_md(report, output_path)

    print(f"\n[OK] Bias analysis complete!")
    print(f"     Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("BIAS ANALYSIS SUMMARY")
    print("=" * 80)

    for level in ['L1', 'L2', 'L3']:
        res = results[level]
        print(f"\n{level}:")
        print(f"  F1 Score Range: {res['min_f1']:.4f} - {res['max_f1']:.4f}")
        print(f"  F1 Std Dev: {res['std_f1']:.4f}")
        print(f"  Categories Below Target (0.90): {res['categories_below_0.90']}/{res['num_categories']}")

        if res['std_f1'] > 0.15:
            print(f"  ⚠️  HIGH VARIANCE - Indicates potential bias")

        if res['categories_below_0.80'] > 0:
            print(f"  ⚠️  {res['categories_below_0.80']} categories critically underperforming")

    print("\n" + "=" * 80)


def generate_bias_report_md(report, output_path):
    """Generate markdown bias analysis report."""

    bias = report['bias_analysis']
    dist = report['distribution']

    md = f"""# Holmes AI - Bias Analysis Report

**Generated:** {report['analysis_date']}

**Model Path:** {report['model_path']}

**Test Dataset:** {report['test_data']}

**Test Samples:** {report['test_samples']:,}

---

## Executive Summary

This report analyzes model performance across all categories to detect potential bias and performance disparities.

### Key Findings

"""

    # Add findings for each level
    for level in ['L1', 'L2', 'L3']:
        res = bias[level]
        md += f"""
#### {level}

- **Categories:** {res['num_categories']}
- **F1 Score Range:** {res['min_f1']:.4f} - {res['max_f1']:.4f}
- **Mean F1:** {res['mean_f1']:.4f} ± {res['std_f1']:.4f}
- **Categories Below 0.90:** {res['categories_below_0.90']}/{res['num_categories']} ({res['categories_below_0.90']/res['num_categories']*100:.1f}%)
- **Categories Below 0.80:** {res['categories_below_0.80']}/{res['num_categories']}
"""

        if res['std_f1'] > 0.15:
            md += f"- ⚠️ **HIGH VARIANCE** ({res['std_f1']:.4f}) indicates potential bias\n"
        else:
            md += f"- ✅ Variance within acceptable range\n"

    md += """
---

## Category Distribution

### Imbalance Analysis

"""

    for level in ['L1', 'L2', 'L3']:
        d = dist[level]
        imbalance_ratio = d['max_samples'] / d['min_samples']
        md += f"""
#### {level}

| Metric | Value |
|--------|-------|
| Total Categories | {d['num_categories']} |
| Min Samples | {d['min_samples']} |
| Max Samples | {d['max_samples']} |
| Mean Samples | {d['mean_samples']:.1f} |
| Std Dev | {d['std_samples']:.1f} |
| **Imbalance Ratio** | **{imbalance_ratio:.2f}x** |

"""
        if imbalance_ratio > 10:
            md += "⚠️ **HIGH IMBALANCE** - Significant disparity between most/least frequent categories\n\n"
        elif imbalance_ratio > 5:
            md += "⚠️ **MODERATE IMBALANCE** - Some categories underrepresented\n\n"
        else:
            md += "✅ Relatively balanced distribution\n\n"

    md += """
---

## Performance vs Frequency Analysis

Plots showing relationship between category frequency and F1 score:

- [L1 Performance vs Frequency](performance_vs_frequency_l1.png)
- [L2 Performance vs Frequency](performance_vs_frequency_l2.png)
- [L3 Performance vs Frequency](performance_vs_frequency_l3.png)

**Interpretation:** If low-frequency categories consistently have lower F1 scores, this indicates sample bias.

---

## Low-Performing Categories

"""

    for level in ['L1', 'L2', 'L3']:
        res = bias[level]
        md += f"\n### {level} Categories Below 0.80 F1\n\n"

        if res['categories_below_0.80'] == 0:
            md += "✅ All categories meet minimum threshold (F1 ≥ 0.80)\n\n"
        else:
            md += f"⚠️ {res['categories_below_0.80']} categories underperforming:\n\n"
            md += "| Category | F1 Score | Samples | Precision | Recall |\n"
            md += "|----------|----------|---------|-----------|--------|\n"

            for cat in res['low_performers'][:10]:  # Top 10 worst
                md += f"| {cat['category']} | {cat['f1_score']:.4f} | {cat['num_samples']} | {cat['precision']:.4f} | {cat['recall']:.4f} |\n"

            md += "\n"

    md += """
---

## Mitigation Recommendations

"""

    # Generate recommendations based on findings
    recommendations = []

    for level in ['L1', 'L2', 'L3']:
        res = bias[level]
        d = dist[level]
        imbalance_ratio = d['max_samples'] / d['min_samples']

        if res['categories_below_0.80'] > 0:
            recommendations.append(f"**{level} Low Performers:** Increase training samples for {res['categories_below_0.80']} underperforming categories to at least 2-3x current count")

        if imbalance_ratio > 10:
            recommendations.append(f"**{level} Class Imbalance:** Implement oversampling/SMOTE for low-frequency categories (bottom 10%)")

        if res['std_f1'] > 0.15:
            recommendations.append(f"**{level} High Variance:** Apply class weighting more aggressively or use focal loss")

        if res['num_low_frequency'] > res['num_categories'] * 0.2:
            recommendations.append(f"**{level} Data Collection:** {res['num_low_frequency']} categories have insufficient samples - prioritize data collection")

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            md += f"{i}. {rec}\n"
    else:
        md += "✅ No critical bias issues detected. Model performance is relatively uniform across categories.\n"

    md += """

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
"""

    with open(output_path / 'BIAS_ANALYSIS_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"[OK] Markdown report saved to {output_path / 'BIAS_ANALYSIS_REPORT.md'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bias analysis for Holmes AI models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved models directory"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test dataset CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bias_analysis",
        help="Output directory for analysis results"
    )

    args = parser.parse_args()
    analyze_bias(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
