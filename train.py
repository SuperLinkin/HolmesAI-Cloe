"""
Training script for Holmes AI models.

This script trains the Sentence-BERT encoder and LightGBM classifier
on labeled transaction data.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data_ingestion import DataIngestion
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier


def load_labeled_data(data_path: str):
    """
    Load labeled transaction data for training.

    Args:
        data_path: Path to labeled data file (CSV or JSON)

    Returns:
        List of transaction dictionaries with labels
    """
    print(f"Loading labeled data from: {data_path}")

    ingestion = DataIngestion()
    normalized = ingestion.ingest_pipeline(data_path)

    # Convert to dictionaries
    transactions = [txn.model_dump() for txn in normalized]

    print(f"Loaded {len(transactions)} labeled transactions")
    return transactions


def train_models(
    data_path: str,
    output_dir: str = "data/models",
    num_boost_round: int = 100,
    validation_split: float = 0.15
):
    """
    Train Sentence-BERT and LightGBM models.

    Args:
        data_path: Path to labeled training data
        output_dir: Directory to save trained models
        num_boost_round: Number of boosting rounds for LightGBM
        validation_split: Fraction of data for validation
    """
    print("\n" + "=" * 80)
    print("HOLMES AI MODEL TRAINING")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    transactions = load_labeled_data(data_path)

    # Step 2: Preprocess
    print("\nPreprocessing transactions...")
    preprocessor = TransactionPreprocessor()
    preprocessed = preprocessor.preprocess_batch(transactions)

    # Step 3: Encode with Sentence-BERT
    print("\nEncoding transactions with Sentence-BERT...")
    encoder = SentenceBERTEncoder()
    embeddings = encoder.encode_transactions(
        preprocessed,
        text_field='merchant_cleaned',
        batch_size=32
    )

    print(f"Generated embeddings: {embeddings.shape}")

    # Step 4: Prepare labels
    print("\nPreparing labels...")
    classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")

    y_l1 = classifier.prepare_labels(preprocessed, level='l1')
    y_l2 = classifier.prepare_labels(preprocessed, level='l2')
    y_l3 = classifier.prepare_labels(preprocessed, level='l3')

    print(f"L1 classes: {len(np.unique(y_l1))}")
    print(f"L2 classes: {len(np.unique(y_l2))}")
    print(f"L3 classes: {len(np.unique(y_l3))}")

    # Step 5: Prepare features
    print("\nPreparing features...")
    X = classifier.prepare_features(embeddings, preprocessed)

    print(f"Feature matrix: {X.shape}")

    # Step 6: Train LightGBM models
    print("\nTraining LightGBM classifiers...")
    scores = classifier.train(
        X, y_l1, y_l2, y_l3,
        validation_split=validation_split,
        num_boost_round=num_boost_round
    )

    # Step 7: Save models
    print("\nSaving models...")

    # Save Sentence-BERT
    encoder_path = output_path / "sentence_bert"
    encoder.save_model(encoder_path)

    # Save LightGBM
    classifier_path = output_path / "lightgbm"
    classifier.save_models(classifier_path)

    # Save training metadata
    metadata = {
        "training_samples": len(transactions),
        "validation_split": validation_split,
        "num_boost_round": num_boost_round,
        "embedding_dimension": embeddings.shape[1],
        "scores": scores,
        "l1_classes": int(len(np.unique(y_l1))),
        "l2_classes": int(len(np.unique(y_l2))),
        "l3_classes": int(len(np.unique(y_l3)))
    }

    with open(output_path / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to: {output_path}")
    print(f"\nPerformance:")
    print(f"  L1 Accuracy: {scores['l1_accuracy']:.4f}")
    print(f"  L2 Accuracy: {scores['l2_accuracy']:.4f}")
    print(f"  L3 Accuracy: {scores['l3_accuracy']:.4f}")

    # Feature importance
    print("\nTop 10 Features (L3 model):")
    feature_importance = classifier.get_feature_importance(level='l3', top_k=10)
    for i, (feature, importance) in enumerate(feature_importance, 1):
        print(f"  {i}. {feature}: {importance:.2f}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train Holmes AI transaction categorization models"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to labeled training data (CSV or JSON)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of boosting rounds for LightGBM"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.15,
        help="Validation split fraction"
    )

    args = parser.parse_args()

    # Train models
    train_models(
        data_path=args.data,
        output_dir=args.output,
        num_boost_round=args.rounds,
        validation_split=args.validation_split
    )


if __name__ == "__main__":
    main()
