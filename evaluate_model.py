"""
Comprehensive Model Evaluation Script

Generates detailed metrics report including:
- Confusion matrices for L1, L2, L3
- Per-class F1 scores
- Macro F1 scores
- Classification reports
- Performance benchmarks (latency, throughput)
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_ingestion import DataIngestion
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier


def plot_confusion_matrix(cm, classes, title, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def evaluate_model(
    model_path: str,
    data_path: str,
    output_dir: str = "evaluation_results"
):
    """
    Evaluate trained models and generate comprehensive metrics report.

    Args:
        model_path: Path to saved models directory
        data_path: Path to test dataset
        output_dir: Directory to save evaluation results
    """

    print("=" * 80)
    print("HOLMES AI - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Step 1: Load test data
    print("[1/8] Loading test data...")
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

    # Step 2: Preprocess
    print("\n[2/8] Preprocessing transactions...")
    preprocessor = TransactionPreprocessor()
    preprocessed = preprocessor.preprocess_batch(transactions)
    print(f"[OK] Preprocessed {len(preprocessed):,} transactions")

    # Step 3: Load models
    print("\n[3/8] Loading trained models...")
    encoder = SentenceBERTEncoder(model_path=f"{model_path}/sentence_bert")
    classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")
    classifier.load_models(f"{model_path}/lightgbm")
    print("[OK] Models loaded successfully")

    # Step 4: Generate embeddings
    print("\n[4/8] Generating embeddings...")
    start_time = time.time()
    embeddings = encoder.encode_transactions(
        preprocessed,
        text_field='merchant_cleaned',
        batch_size=64
    )
    embedding_time = time.time() - start_time
    print(f"[OK] Generated {embeddings.shape[0]:,} embeddings in {embedding_time:.2f}s")

    # Step 5: Prepare features
    print("\n[5/8] Preparing features...")
    X = classifier.prepare_features(embeddings, preprocessed, include_enrichment=True)
    print(f"[OK] Feature matrix: {X.shape}")

    # Step 6: Make predictions and measure latency
    print("\n[6/8] Making predictions and measuring performance...")

    # Batch prediction
    start_time = time.time()
    predictions = classifier.predict(X, use_hierarchy=True)
    batch_time = time.time() - start_time

    # Individual prediction latency (sample 1000 random transactions)
    sample_indices = np.random.choice(len(X), min(1000, len(X)), replace=False)
    latencies = []

    for idx in sample_indices:
        start = time.time()
        _ = classifier.predict(X[idx:idx+1], use_hierarchy=True)
        latencies.append(time.time() - start)

    avg_latency_ms = np.mean(latencies) * 1000
    p95_latency_ms = np.percentile(latencies, 95) * 1000
    p99_latency_ms = np.percentile(latencies, 99) * 1000
    throughput = len(X) / batch_time

    print(f"[OK] Predictions complete")
    print(f"     Average latency: {avg_latency_ms:.2f}ms")
    print(f"     P95 latency: {p95_latency_ms:.2f}ms")
    print(f"     P99 latency: {p99_latency_ms:.2f}ms")
    print(f"     Throughput: {throughput:.0f} txns/sec")

    # Step 7: Calculate metrics
    print("\n[7/8] Calculating evaluation metrics...")

    # Extract ground truth
    y_true_l1 = [t['l1'] for t in preprocessed]
    y_true_l2 = [t['l2'] for t in preprocessed]
    y_true_l3 = [t['l3'] for t in preprocessed]

    # Extract predictions
    y_pred_l1 = predictions['l1']
    y_pred_l2 = predictions['l2']
    y_pred_l3 = predictions['l3']

    # Calculate metrics for each level
    metrics = {}

    for level, y_true, y_pred in [
        ('L1', y_true_l1, y_pred_l1),
        ('L2', y_true_l2, y_pred_l2),
        ('L3', y_true_l3, y_pred_l3)
    ]:
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(list(set(y_true) | set(y_pred)))

        # Save confusion matrix plot
        plot_confusion_matrix(
            cm,
            classes,
            f'{level} Confusion Matrix',
            output_path / f'confusion_matrix_{level.lower()}.png'
        )

        metrics[level] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'num_classes': len(classes),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        print(f"\n{level} Metrics:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"  Classes: {len(classes)}")

    # Step 8: Generate comprehensive report
    print("\n[8/8] Generating evaluation report...")

    report = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'model_path': model_path,
        'test_data': data_path,
        'test_samples': len(transactions),

        'performance': {
            'avg_latency_ms': round(avg_latency_ms, 2),
            'p95_latency_ms': round(p95_latency_ms, 2),
            'p99_latency_ms': round(p99_latency_ms, 2),
            'throughput_txns_per_sec': round(throughput, 2),
            'embedding_time_sec': round(embedding_time, 2)
        },

        'metrics': {
            'L1': {
                'accuracy': round(metrics['L1']['accuracy'], 4),
                'macro_f1': round(metrics['L1']['macro_f1'], 4),
                'weighted_f1': round(metrics['L1']['weighted_f1'], 4),
                'num_classes': metrics['L1']['num_classes']
            },
            'L2': {
                'accuracy': round(metrics['L2']['accuracy'], 4),
                'macro_f1': round(metrics['L2']['macro_f1'], 4),
                'weighted_f1': round(metrics['L2']['weighted_f1'], 4),
                'num_classes': metrics['L2']['num_classes']
            },
            'L3': {
                'accuracy': round(metrics['L3']['accuracy'], 4),
                'macro_f1': round(metrics['L3']['macro_f1'], 4),
                'weighted_f1': round(metrics['L3']['weighted_f1'], 4),
                'num_classes': metrics['L3']['num_classes']
            }
        },

        'target_achievement': {
            'L1_meets_target': metrics['L1']['macro_f1'] >= 0.90,
            'L2_meets_target': metrics['L2']['macro_f1'] >= 0.90,
            'L3_meets_target': metrics['L3']['macro_f1'] >= 0.90,
            'overall_meets_target': all([
                metrics['L1']['macro_f1'] >= 0.90,
                metrics['L2']['macro_f1'] >= 0.90,
                metrics['L3']['macro_f1'] >= 0.90
            ])
        }
    }

    # Save JSON report
    with open(output_path / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Save detailed classification reports
    for level in ['L1', 'L2', 'L3']:
        report_df = pd.DataFrame(metrics[level]['classification_report']).transpose()
        report_df.to_csv(output_path / f'classification_report_{level.lower()}.csv')

    # Generate markdown report
    generate_markdown_report(report, metrics, output_path)

    print(f"\n[OK] Evaluation complete!")
    print(f"     Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nTest Samples: {len(transactions):,}")
    print(f"\nPerformance:")
    print(f"  Average Latency: {avg_latency_ms:.2f}ms")
    print(f"  Throughput: {throughput:.0f} txns/sec")
    print(f"\nMacro F1 Scores:")
    print(f"  L1: {metrics['L1']['macro_f1']:.4f} {'✅' if metrics['L1']['macro_f1'] >= 0.90 else '❌'}")
    print(f"  L2: {metrics['L2']['macro_f1']:.4f} {'✅' if metrics['L2']['macro_f1'] >= 0.90 else '❌'}")
    print(f"  L3: {metrics['L3']['macro_f1']:.4f} {'✅' if metrics['L3']['macro_f1'] >= 0.90 else '❌'}")
    print(f"\nTarget (Macro F1 ≥ 0.90): {'✅ ACHIEVED' if report['target_achievement']['overall_meets_target'] else '❌ NOT MET'}")
    print("=" * 80)


def generate_markdown_report(report, metrics, output_path):
    """Generate markdown evaluation report."""

    md = f"""# Holmes AI - Model Evaluation Report

**Generated:** {report['evaluation_date']}

**Model Path:** {report['model_path']}

**Test Dataset:** {report['test_data']}

**Test Samples:** {report['test_samples']:,}

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Average Latency | {report['performance']['avg_latency_ms']:.2f}ms |
| P95 Latency | {report['performance']['p95_latency_ms']:.2f}ms |
| P99 Latency | {report['performance']['p99_latency_ms']:.2f}ms |
| Throughput | {report['performance']['throughput_txns_per_sec']:.0f} txns/sec |
| Embedding Time | {report['performance']['embedding_time_sec']:.2f}s |

---

## Accuracy Metrics

### Summary

| Level | Accuracy | Macro F1 | Weighted F1 | Classes | Target (≥0.90) |
|-------|----------|----------|-------------|---------|----------------|
| **L1** | {metrics['L1']['accuracy']:.4f} ({metrics['L1']['accuracy']*100:.2f}%) | **{metrics['L1']['macro_f1']:.4f}** | {metrics['L1']['weighted_f1']:.4f} | {metrics['L1']['num_classes']} | {'✅' if metrics['L1']['macro_f1'] >= 0.90 else '❌'} |
| **L2** | {metrics['L2']['accuracy']:.4f} ({metrics['L2']['accuracy']*100:.2f}%) | **{metrics['L2']['macro_f1']:.4f}** | {metrics['L2']['weighted_f1']:.4f} | {metrics['L2']['num_classes']} | {'✅' if metrics['L2']['macro_f1'] >= 0.90 else '❌'} |
| **L3** | {metrics['L3']['accuracy']:.4f} ({metrics['L3']['accuracy']*100:.2f}%) | **{metrics['L3']['macro_f1']:.4f}** | {metrics['L3']['weighted_f1']:.4f} | {metrics['L3']['num_classes']} | {'✅' if metrics['L3']['macro_f1'] >= 0.90 else '❌'} |

### Target Achievement

**Overall Target (All levels Macro F1 ≥ 0.90):** {'✅ **ACHIEVED**' if report['target_achievement']['overall_meets_target'] else '❌ **NOT MET**'}

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

"""

    if report['target_achievement']['overall_meets_target']:
        md += """
✅ **All levels meet the target Macro F1 ≥ 0.90**

The model is production-ready with excellent performance across all hierarchical levels.
"""
    else:
        md += """
⚠️ **Some levels do not meet the target Macro F1 ≥ 0.90**

### Suggested Improvements:

"""
        if not report['target_achievement']['L1_meets_target']:
            md += f"- **L1** (F1: {metrics['L1']['macro_f1']:.4f}): Increase training data or adjust L1 hyperparameters\n"
        if not report['target_achievement']['L2_meets_target']:
            md += f"- **L2** (F1: {metrics['L2']['macro_f1']:.4f}): Consider training on 150k-200k samples\n"
        if not report['target_achievement']['L3_meets_target']:
            md += f"- **L3** (F1: {metrics['L3']['macro_f1']:.4f}): May need 200k-500k samples for granular L3 categories\n"

    md += """

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
"""

    with open(output_path / 'EVALUATION_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"[OK] Markdown report saved to {output_path / 'EVALUATION_REPORT.md'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation for Holmes AI"
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
        default="evaluation_results",
        help="Output directory for evaluation results"
    )

    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
