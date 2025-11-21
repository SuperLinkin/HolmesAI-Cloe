"""
Inference script for Holmes AI transaction categorization.

Run categorization on new transaction data using trained models.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

from src.data_ingestion import DataIngestion
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier
from src.utils import ConfidenceScorer


class TransactionCategorizer:
    """
    Main categorization pipeline combining all components.
    """

    def __init__(
        self,
        encoder_path: str,
        classifier_path: str,
        taxonomy_path: str = "src/config/taxonomy.json"
    ):
        """
        Initialize categorizer with trained models.

        Args:
            encoder_path: Path to trained Sentence-BERT model
            classifier_path: Path to trained LightGBM models
            taxonomy_path: Path to taxonomy.json
        """
        print("Loading Holmes AI models...")

        # Load preprocessor
        self.preprocessor = TransactionPreprocessor()

        # Load encoder
        self.encoder = SentenceBERTEncoder()
        self.encoder.load_model(encoder_path)

        # Load classifier
        self.classifier = LightGBMClassifier(taxonomy_path=taxonomy_path)
        self.classifier.load_models(classifier_path)

        # Load confidence scorer
        self.confidence_scorer = ConfidenceScorer(taxonomy_path=taxonomy_path)

        print("Models loaded successfully!")

    def categorize_transactions(
        self,
        transactions: List[Dict]
    ) -> List[Dict]:
        """
        Categorize a batch of transactions.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of categorized transactions with predictions
        """
        start_time = time.time()

        # Step 1: Preprocess
        preprocessed = self.preprocessor.preprocess_batch(transactions)

        # Step 2: Encode
        embeddings = self.encoder.encode_transactions(
            preprocessed,
            text_field='merchant_cleaned',
            batch_size=32
        )

        # Step 3: Prepare features
        X = self.classifier.prepare_features(embeddings, preprocessed)

        # Step 4: Predict
        predictions = self.classifier.predict(X, return_proba=True)

        # Step 5: Build results with confidence scores
        results = []

        for i, txn in enumerate(preprocessed):
            # Get predicted indices
            l3_proba = predictions['l3_proba'][i]
            l3_pred_idx = l3_proba.argmax()

            # Get predicted category IDs
            l1 = predictions['l1'][i]
            l2 = predictions['l2'][i]
            l3 = predictions['l3'][i]

            # Extract category IDs from taxonomy
            l1_id, l2_id, l3_id = self._get_category_ids(l1, l2, l3)

            # Calculate confidence
            confidence_result = self.confidence_scorer.score_prediction(
                transaction=txn,
                l3_probabilities=l3_proba,
                predicted_l3_idx=l3_pred_idx,
                predicted_l3_id=l3_id,
                predicted_l2_id=l2_id
            )

            # Build result
            result = {
                'transaction_id': txn['transaction_id'],
                'merchant_raw': txn['merchant_raw'],
                'merchant_cleaned': txn['merchant_cleaned'],
                'amount': txn['amount'],
                'category': {
                    'l1': l1,
                    'l1_id': l1_id,
                    'l2': l2,
                    'l2_id': l2_id,
                    'l3': l3,
                    'l3_id': l3_id
                },
                'confidence': confidence_result['confidence'],
                'confidence_level': confidence_result['confidence_level'],
                'should_review': confidence_result['should_review'],
                'confidence_components': confidence_result['components']
            }

            results.append(result)

        total_time = time.time() - start_time
        avg_time_ms = (total_time * 1000) / len(transactions)

        print(f"\nCategorized {len(transactions)} transactions")
        print(f"Average processing time: {avg_time_ms:.2f} ms/transaction")
        print(f"Total time: {total_time:.2f} seconds")

        return results

    def _get_category_ids(self, l1: str, l2: str, l3: str) -> tuple:
        """
        Extract category IDs from taxonomy.

        Args:
            l1: L1 category name
            l2: L2 category name
            l3: L3 category name

        Returns:
            Tuple of (l1_id, l2_id, l3_id)
        """
        taxonomy = self.confidence_scorer.taxonomy

        for l1_cat in taxonomy['categories']:
            if l1_cat['l1'] == l1:
                l1_id = l1_cat['l1_id']

                for l2_cat in l1_cat['l2_subcategories']:
                    if l2_cat['l2'] == l2:
                        l2_id = l2_cat['l2_id']

                        for l3_cat in l2_cat['l3_types']:
                            if l3_cat['l3'] == l3:
                                l3_id = l3_cat['l3_id']
                                return l1_id, l2_id, l3_id

        # Fallback
        return "UNK", "UNK", "UNK"


def run_inference(
    data_path: str,
    encoder_path: str,
    classifier_path: str,
    output_path: str = None
):
    """
    Run inference on transaction data.

    Args:
        data_path: Path to transaction data file
        encoder_path: Path to trained encoder
        classifier_path: Path to trained classifier
        output_path: Path to save results (optional)
    """
    print("\n" + "=" * 80)
    print("HOLMES AI INFERENCE")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {data_path}")
    ingestion = DataIngestion()
    normalized = ingestion.ingest_pipeline(data_path)
    transactions = [txn.model_dump() for txn in normalized]

    # Initialize categorizer
    categorizer = TransactionCategorizer(
        encoder_path=encoder_path,
        classifier_path=classifier_path
    )

    # Categorize
    print("\nCategorizing transactions...")
    results = categorizer.categorize_transactions(transactions)

    # Statistics
    low_confidence_count = sum(1 for r in results if r['should_review'])
    avg_confidence = sum(r['confidence'] for r in results) / len(results)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total transactions: {len(results)}")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"Low confidence (review needed): {low_confidence_count}")

    # Show sample results
    print("\nSample results:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. {result['merchant_raw']}")
        print(f"   Cleaned: {result['merchant_cleaned']}")
        print(f"   Category: {result['category']['l3']}")
        print(f"   Confidence: {result['confidence']:.4f} ({result['confidence_level']})")
        if result['should_review']:
            print(f"   ⚠️  NEEDS REVIEW")

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(
        description="Run Holmes AI transaction categorization inference"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to transaction data (CSV or JSON)"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="data/models/sentence_bert",
        help="Path to trained Sentence-BERT model"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="data/models/lightgbm",
        help="Path to trained LightGBM models"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results (optional)"
    )

    args = parser.parse_args()

    # Run inference
    run_inference(
        data_path=args.data,
        encoder_path=args.encoder,
        classifier_path=args.classifier,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
