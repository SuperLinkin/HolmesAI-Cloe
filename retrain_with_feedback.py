"""
Retraining Pipeline with User Feedback

This script retrains the Holmes AI models incorporating user feedback corrections.
It combines the original training data with feedback data for continuous improvement.
"""

import argparse
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.feedback import FeedbackStorage
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_feedback_data(
    feedback_storage: FeedbackStorage,
    min_samples: int = 100,
    unused_only: bool = True
) -> pd.DataFrame:
    """
    Load feedback data for retraining.

    Args:
        feedback_storage: Feedback storage instance
        min_samples: Minimum feedback samples required
        unused_only: Only use feedback not yet used in training

    Returns:
        DataFrame with feedback data
    """
    print(f"\n=== Loading Feedback Data ===")
    feedback_df = feedback_storage.get_feedback_for_training(
        min_samples=min_samples,
        unused_only=unused_only
    )

    print(f"Loaded {len(feedback_df)} feedback samples")
    if unused_only:
        print(f"(Unused feedback only)")

    return feedback_df


def combine_datasets(
    original_data_path: str,
    feedback_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine original training data with feedback data.

    Args:
        original_data_path: Path to original training CSV
        feedback_df: DataFrame with feedback data

    Returns:
        Combined DataFrame
    """
    print(f"\n=== Combining Datasets ===")

    # Load original data
    original_df = pd.read_csv(original_data_path)
    print(f"Original dataset: {len(original_df)} samples")

    # Remove feedback_id column if present
    if 'feedback_id' in feedback_df.columns:
        feedback_ids = feedback_df['feedback_id'].tolist()
        feedback_df = feedback_df.drop(columns=['feedback_id'])
    else:
        feedback_ids = []

    print(f"Feedback dataset: {len(feedback_df)} samples")

    # Combine datasets
    combined_df = pd.concat([original_df, feedback_df], ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} samples")

    return combined_df, feedback_ids


def retrain_models(
    combined_df: pd.DataFrame,
    output_dir: str,
    validation_split: float = 0.15,
    n_estimators: int = 500,
    verbose: bool = True
) -> dict:
    """
    Retrain models with combined dataset.

    Args:
        combined_df: Combined training data
        output_dir: Output directory for models
        validation_split: Validation split ratio
        n_estimators: Number of LightGBM trees
        verbose: Print progress

    Returns:
        Dictionary with training metrics
    """
    print(f"\n=== Retraining Models ===")
    start_time = time.time()

    # Split data
    train_df, val_df = train_test_split(
        combined_df,
        test_size=validation_split,
        stratify=combined_df['L1'],
        random_state=42
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Preprocess
    print("\n1. Preprocessing transactions...")
    preprocessor = TransactionPreprocessor()
    train_preprocessed = preprocessor.preprocess_batch(train_df.to_dict('records'))
    val_preprocessed = preprocessor.preprocess_batch(val_df.to_dict('records'))

    # Generate embeddings
    print("\n2. Generating embeddings...")
    encoder = SentenceBERTEncoder(model_name='all-mpnet-base-v2')

    train_merchants = [t['merchant_cleaned'] for t in train_preprocessed]
    val_merchants = [t['merchant_cleaned'] for t in val_preprocessed]

    train_embeddings = encoder.encode_batch(train_merchants)
    val_embeddings = encoder.encode_batch(val_merchants)

    print(f"   Train embeddings: {train_embeddings.shape}")
    print(f"   Validation embeddings: {val_embeddings.shape}")

    # Extract features
    print("\n3. Extracting engineered features...")
    train_features = preprocessor.extract_features_batch(train_preprocessed, train_embeddings)
    val_features = preprocessor.extract_features_batch(val_preprocessed, val_embeddings)

    # Extract labels
    train_labels = {
        'L1': [t['L1'] for t in train_preprocessed],
        'L2': [t['L2'] for t in train_preprocessed],
        'L3': [t['L3'] for t in train_preprocessed]
    }
    val_labels = {
        'L1': [t['L1'] for t in val_preprocessed],
        'L2': [t['L2'] for t in val_preprocessed],
        'L3': [t['L3'] for t in val_preprocessed]
    }

    # Train classifier
    print("\n4. Training LightGBM classifiers...")
    classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")

    classifier.train(
        train_features,
        train_labels,
        val_features,
        val_labels,
        n_estimators=n_estimators,
        verbose=verbose
    )

    # Evaluate
    print("\n5. Evaluating on validation set...")
    val_predictions = classifier.predict(val_features)

    metrics = {}
    for level in ['L1', 'L2', 'L3']:
        accuracy = accuracy_score(val_labels[level], val_predictions[level])
        f1_macro = f1_score(val_labels[level], val_predictions[level], average='macro')

        metrics[f'{level}_accuracy'] = accuracy
        metrics[f'{level}_f1'] = f1_macro

        print(f"\n{level} Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {f1_macro:.4f}")

    # Save models
    print(f"\n6. Saving models to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    encoder_path = Path(output_dir) / "sentence_bert"
    classifier_path = Path(output_dir) / "lightgbm"

    encoder.save(str(encoder_path))
    classifier.save(str(classifier_path))

    print("[OK] Models saved successfully")

    training_duration = time.time() - start_time
    metrics['training_duration_sec'] = training_duration
    metrics['training_samples'] = len(train_df)
    metrics['validation_samples'] = len(val_df)

    print(f"\n=== Retraining Complete ===")
    print(f"Total time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Retrain Holmes AI models with user feedback"
    )
    parser.add_argument(
        '--original-data',
        type=str,
        required=True,
        help='Path to original training data CSV'
    )
    parser.add_argument(
        '--feedback-db',
        type=str,
        default='data/feedback.db',
        help='Path to feedback database'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models_retrained',
        help='Output directory for retrained models'
    )
    parser.add_argument(
        '--min-feedback',
        type=int,
        default=100,
        help='Minimum feedback samples required for retraining'
    )
    parser.add_argument(
        '--unused-only',
        action='store_true',
        help='Only use feedback not yet used in training'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=500,
        help='Number of LightGBM trees (default: 500)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Holmes AI - Retraining with User Feedback")
    print("="*60)

    # Generate run ID
    run_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nRun ID: {run_id}")

    # Initialize feedback storage
    feedback_storage = FeedbackStorage(db_path=args.feedback_db)

    # Check feedback count
    feedback_count = feedback_storage.get_feedback_count(unused_only=args.unused_only)
    print(f"\nAvailable feedback: {feedback_count} samples")

    if feedback_count < args.min_feedback:
        print(f"\n[WARN] Insufficient feedback samples for retraining")
        print(f"       Required: {args.min_feedback}, Available: {feedback_count}")
        print(f"       Collect more feedback before retraining.")
        return

    # Load feedback data
    feedback_df = load_feedback_data(
        feedback_storage,
        min_samples=args.min_feedback,
        unused_only=args.unused_only
    )

    # Combine with original data
    combined_df, feedback_ids = combine_datasets(args.original_data, feedback_df)

    # Record retraining start
    retraining_id = feedback_storage.add_retraining_run(
        run_id=run_id,
        feedback_count=len(feedback_df),
        training_samples=len(combined_df),
        validation_samples=int(len(combined_df) * args.validation_split),
        config={
            'original_data': args.original_data,
            'feedback_count': len(feedback_df),
            'n_estimators': args.n_estimators,
            'validation_split': args.validation_split
        }
    )

    try:
        # Retrain models
        metrics = retrain_models(
            combined_df,
            output_dir=args.output,
            validation_split=args.validation_split,
            n_estimators=args.n_estimators
        )

        # Update retraining history
        feedback_storage.update_retraining_run(
            run_id=run_id,
            l1_accuracy=metrics['L1_accuracy'],
            l2_accuracy=metrics['L2_accuracy'],
            l3_accuracy=metrics['L3_accuracy'],
            l1_f1=metrics['L1_f1'],
            l2_f1=metrics['L2_f1'],
            l3_f1=metrics['L3_f1'],
            training_duration_sec=metrics['training_duration_sec'],
            model_path=args.output,
            status='completed'
        )

        # Mark feedback as used
        if feedback_ids:
            feedback_storage.mark_feedback_as_used(feedback_ids, run_id)

        print("\n" + "="*60)
        print("Retraining Successful!")
        print("="*60)
        print(f"\nNew models saved to: {args.output}")
        print(f"\nPerformance Summary:")
        print(f"  L1: Accuracy={metrics['L1_accuracy']:.4f}, F1={metrics['L1_f1']:.4f}")
        print(f"  L2: Accuracy={metrics['L2_accuracy']:.4f}, F1={metrics['L2_f1']:.4f}")
        print(f"  L3: Accuracy={metrics['L3_accuracy']:.4f}, F1={metrics['L3_f1']:.4f}")
        print(f"\n{len(feedback_df)} feedback samples incorporated into training.")
        print(f"Total training samples: {metrics['training_samples']}")

    except Exception as e:
        # Mark retraining as failed
        feedback_storage.update_retraining_run(
            run_id=run_id,
            status='failed'
        )
        print(f"\n[ERROR] Retraining failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
