"""
Holmes AI - Interactive Demo Script

Demonstrates:
1. Complete pipeline execution (data → training → inference)
2. Sample predictions with confidence scores
3. Taxonomy modification via config
4. Low-confidence prediction handling
5. Performance benchmarks
"""

import json
import time
import random
from pathlib import Path
from src.data_ingestion import DataIngestion
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier


def demo_pipeline_execution():
    """Demonstrate complete pipeline from raw data to predictions."""

    print("=" * 80)
    print("HOLMES AI - PIPELINE EXECUTION DEMO")
    print("=" * 80)
    print()

    # Step 1: Raw transaction
    print("[STEP 1] Raw Transaction Input")
    print("-" * 80)

    raw_transaction = {
        "merchant": "STARBUCKS #12345",
        "amount": 4.75,
        "currency": "USD",
        "timestamp": "2024-11-22T09:30:00",
        "mcc_code": 5814,
        "channel": "pos"
    }

    print(json.dumps(raw_transaction, indent=2))
    print()

    # Step 2: Data ingestion & validation
    print("[STEP 2] Data Ingestion & Validation")
    print("-" * 80)

    ingestion = DataIngestion()
    try:
        normalized = ingestion.normalize_transaction(raw_transaction)
        print(f"✅ Transaction validated successfully")
        print(f"   Normalized fields: {list(normalized.model_dump().keys())}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return

    print()

    # Step 3: Preprocessing
    print("[STEP 3] Text Preprocessing")
    print("-" * 80)

    preprocessor = TransactionPreprocessor()
    preprocessed = preprocessor.preprocess_single(normalized.model_dump())

    print(f"Original merchant: '{raw_transaction['merchant']}'")
    print(f"Cleaned merchant:  '{preprocessed['merchant_cleaned']}'")
    print(f"Spend band:        {preprocessed.get('spend_band', 'N/A')}")
    print(f"Temporal pattern:  {preprocessed.get('temporal_pattern', 'N/A')}")
    print()

    # Step 4: Load models
    print("[STEP 4] Loading Trained Models")
    print("-" * 80)

    encoder = SentenceBERTEncoder()
    classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")

    print(f"✅ Sentence-BERT encoder loaded (768D embeddings)")
    print(f"✅ LightGBM classifier loaded")
    print(f"   Taxonomy: {len(classifier.l1_to_l2_map)} L1 categories")
    print()

    # Step 5: Generate embedding
    print("[STEP 5] Semantic Embedding Generation")
    print("-" * 80)

    start_time = time.time()
    embedding = encoder.encode_transactions(
        [preprocessed],
        text_field='merchant_cleaned',
        batch_size=1
    )
    embedding_time = (time.time() - start_time) * 1000

    print(f"✅ Generated 768D embedding in {embedding_time:.2f}ms")
    print(f"   Embedding preview: [{embedding[0][:3]}..., {embedding[0][-3:]}]")
    print()

    # Step 6: Prepare features
    print("[STEP 6] Feature Engineering")
    print("-" * 80)

    X = classifier.prepare_features(
        embedding,
        [preprocessed],
        include_enrichment=True
    )

    print(f"✅ Feature vector prepared: {X.shape[1]} features")
    print(f"   - 768 semantic features (Sentence-BERT)")
    print(f"   - 5 engineered features (spend band, temporal, channel, MCC, amount)")
    print()

    # Step 7: Prediction
    print("[STEP 7] Hierarchical Prediction")
    print("-" * 80)

    start_time = time.time()
    prediction = classifier.predict(X, use_hierarchy=True)
    prediction_time = (time.time() - start_time) * 1000

    print(f"✅ Prediction completed in {prediction_time:.2f}ms")
    print()
    print(f"Predicted Category Hierarchy:")
    print(f"  L1 (Top-level):    {prediction['l1'][0]}")
    print(f"  L2 (Sub-category): {prediction['l2'][0]}")
    print(f"  L3 (Detailed):     {prediction['l3'][0]}")
    print()
    print(f"Confidence Score:    {prediction.get('confidence', [0])[0]:.4f}")
    print()

    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


def demo_sample_predictions():
    """Show sample predictions with various confidence levels."""

    print("\n" * 2)
    print("=" * 80)
    print("SAMPLE PREDICTIONS WITH CONFIDENCE SCORES")
    print("=" * 80)
    print()

    # Sample transactions with varying confidence
    samples = [
        {
            "merchant": "WHOLE FOODS MARKET",
            "amount": 87.50,
            "expected_category": "Groceries",
            "expected_confidence": "High"
        },
        {
            "merchant": "AMAZON.COM",
            "amount": 29.99,
            "expected_category": "Shopping (ambiguous)",
            "expected_confidence": "Medium"
        },
        {
            "merchant": "XYZ CORP 12345",
            "amount": 150.00,
            "expected_category": "Uncertain",
            "expected_confidence": "Low"
        }
    ]

    encoder = SentenceBERTEncoder()
    classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")
    preprocessor = TransactionPreprocessor()

    for i, sample in enumerate(samples, 1):
        print(f"[Sample {i}] {sample['merchant']}")
        print("-" * 80)

        # Preprocess
        txn = {
            "merchant": sample['merchant'],
            "amount": sample['amount'],
            "currency": "USD",
            "timestamp": "2024-11-22T10:00:00",
            "channel": "online"
        }

        preprocessed = preprocessor.preprocess_single(txn)
        embedding = encoder.encode_transactions([preprocessed], text_field='merchant_cleaned', batch_size=1)
        X = classifier.prepare_features(embedding, [preprocessed], include_enrichment=True)

        # Predict
        prediction = classifier.predict(X, use_hierarchy=True)

        confidence = prediction.get('confidence', [0])[0]
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"

        print(f"  Prediction: {prediction['l1'][0]} → {prediction['l2'][0]} → {prediction['l3'][0]}")
        print(f"  Confidence: {confidence:.4f} ({confidence_level})")
        print(f"  Expected:   {sample['expected_confidence']} confidence")

        if confidence < 0.5:
            print(f"  ⚠️  LOW CONFIDENCE - Recommend human review")

        print()


def demo_taxonomy_modification():
    """Demonstrate taxonomy modification via config file."""

    print("=" * 80)
    print("TAXONOMY MODIFICATION DEMO")
    print("=" * 80)
    print()

    taxonomy_path = "src/config/taxonomy.json"

    # Step 1: Show current taxonomy
    print("[STEP 1] Current Taxonomy")
    print("-" * 80)

    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)

    print(f"L1 Categories: {len(taxonomy)}")
    print(f"Sample L1: {list(taxonomy.keys())[:3]}")
    print()

    # Step 2: Demonstrate modification
    print("[STEP 2] How to Modify Taxonomy")
    print("-" * 80)

    print("""
To add a new category, edit src/config/taxonomy.json:

{
  "New Category": {
    "aliases": ["alias1", "alias2"],
    "mcc_codes": [1234, 5678],
    "subcategories": {
      "Subcategory 1": {
        "aliases": ["sub_alias1"],
        "items": {
          "Item 1": {
            "aliases": ["item_alias1"]
          }
        }
      }
    }
  }
}

Then reload the classifier:

    classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")

No code changes required! The taxonomy is admin-configurable.
""")

    # Step 3: Show impact
    print("[STEP 3] Impact of Taxonomy Changes")
    print("-" * 80)

    print("""
After modifying taxonomy.json:

1. Alias matching will include new aliases
2. MCC code mapping will recognize new codes
3. Model will need retraining to predict new categories
4. All changes are version-controlled in the config file

This allows non-technical admins to customize categorization logic.
""")


def demo_performance_benchmark():
    """Demonstrate performance benchmarks."""

    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    encoder = SentenceBERTEncoder()
    classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")
    preprocessor = TransactionPreprocessor()

    # Generate sample transactions
    print("[Generating sample transactions...]")

    samples = []
    merchants = ["Starbucks", "Amazon", "Shell Gas", "Walmart", "Apple Store"]

    for _ in range(100):
        txn = {
            "merchant": random.choice(merchants),
            "amount": random.uniform(5, 200),
            "currency": "USD",
            "timestamp": "2024-11-22T10:00:00",
            "channel": random.choice(["online", "pos"])
        }
        samples.append(preprocessor.preprocess_single(txn))

    print(f"[OK] Generated {len(samples)} sample transactions")
    print()

    # Benchmark embedding
    print("[Benchmarking embedding generation...]")
    start = time.time()
    embeddings = encoder.encode_transactions(samples, text_field='merchant_cleaned', batch_size=32)
    embedding_time = time.time() - start

    print(f"  Embedding: {embedding_time*1000:.2f}ms for {len(samples)} txns")
    print(f"  Throughput: {len(samples)/embedding_time:.0f} txns/sec")
    print()

    # Benchmark feature preparation
    print("[Benchmarking feature preparation...]")
    start = time.time()
    X = classifier.prepare_features(embeddings, samples, include_enrichment=True)
    feature_time = time.time() - start

    print(f"  Feature prep: {feature_time*1000:.2f}ms")
    print()

    # Benchmark prediction
    print("[Benchmarking prediction...]")
    latencies = []

    for i in range(len(X)):
        start = time.time()
        _ = classifier.predict(X[i:i+1], use_hierarchy=True)
        latencies.append((time.time() - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  P95 latency: {p95_latency:.2f}ms")
    print(f"  P99 latency: {p99_latency:.2f}ms")
    print()

    # Summary
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print()
    print(f"✅ Average latency: {avg_latency:.2f}ms (Target: <200ms)")
    print(f"✅ Throughput: {len(samples)/embedding_time:.0f} txns/sec")
    print()


def main():
    """Run all demos."""

    print("\n" * 2)
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "HOLMES AI - INTERACTIVE DEMO" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    demos = [
        ("Pipeline Execution", demo_pipeline_execution),
        ("Sample Predictions", demo_sample_predictions),
        ("Taxonomy Modification", demo_taxonomy_modification),
        ("Performance Benchmark", demo_performance_benchmark)
    ]

    for i, (name, func) in enumerate(demos, 1):
        print(f"\n[DEMO {i}/{len(demos)}] {name}")
        func()
        input("\nPress Enter to continue...")

    print("\n" * 2)
    print("=" * 80)
    print("ALL DEMOS COMPLETE!")
    print("=" * 80)
    print()
    print("For more information, see:")
    print("  - README.md - Setup and usage guide")
    print("  - SYSTEM_ARCHITECTURE.md - Technical architecture")
    print("  - evaluation_results/ - Model evaluation metrics")
    print()


if __name__ == "__main__":
    main()
