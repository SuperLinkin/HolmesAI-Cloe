"""
Main preprocessing pipeline combining text cleaning and feature enrichment.
"""

from typing import List, Dict
from .text_cleaner import TextCleaner
from .feature_enrichment import FeatureEnrichment


class TransactionPreprocessor:
    """
    Main preprocessing pipeline for transaction data.
    Combines text cleaning and feature enrichment.
    """

    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.feature_enricher = FeatureEnrichment()

    def preprocess_transaction(self, transaction: Dict) -> Dict:
        """
        Preprocess a single transaction.

        Args:
            transaction: Raw transaction dictionary

        Returns:
            Preprocessed transaction with cleaned text and enriched features
        """
        # Step 1: Clean merchant name
        if 'merchant_raw' in transaction:
            transaction['merchant_cleaned'] = self.text_cleaner.clean(
                transaction['merchant_raw']
            )
            transaction['merchant_tokens'] = self.text_cleaner.extract_tokens(
                transaction['merchant_cleaned']
            )

        # Step 2: Enrich with derived features
        enriched = self.feature_enricher.enrich_transaction(transaction)

        return enriched

    def preprocess_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Preprocess a batch of transactions.

        Args:
            transactions: List of raw transaction dictionaries

        Returns:
            List of preprocessed transactions
        """
        return [self.preprocess_transaction(txn) for txn in transactions]

    def get_feature_summary(self, transaction: Dict) -> Dict:
        """
        Get a summary of extracted features for debugging/explainability.

        Args:
            transaction: Preprocessed transaction

        Returns:
            Dictionary with feature summary
        """
        return {
            'merchant_raw': transaction.get('merchant_raw', ''),
            'merchant_cleaned': transaction.get('merchant_cleaned', ''),
            'tokens': transaction.get('merchant_tokens', []),
            'spend_band': transaction.get('spend_band', ''),
            'temporal_pattern': transaction.get('temporal_pattern', ''),
            'city': transaction.get('city', ''),
            'region': transaction.get('region', ''),
            'mcc_category_group': transaction.get('mcc_category_group', ''),
            'is_refund': transaction.get('is_refund', False),
            'is_weekend': transaction.get('is_weekend', False)
        }


def main():
    """Example usage of TransactionPreprocessor."""

    preprocessor = TransactionPreprocessor()

    # Test transactions
    test_transactions = [
        {
            'transaction_id': 'TXN_001',
            'merchant_raw': 'AMZN MKTP US*2A3B4C5D6 SEATTLE WA',
            'amount': 49.99,
            'currency': 'USD',
            'day_of_week': 0,
            'hour_of_day': 14,
            'month': 1,
            'location': 'Seattle, WA',
            'mcc_code': '5942'
        },
        {
            'transaction_id': 'TXN_002',
            'merchant_raw': 'SWIGGY*FOOD DELIVERY',
            'amount': 25.50,
            'currency': 'USD',
            'day_of_week': 5,
            'hour_of_day': 20,
            'month': 1,
            'location': 'Bangalore, KA',
            'mcc_code': '5814'
        }
    ]

    print("Preprocessing Pipeline Example:")
    print("=" * 80)

    preprocessed = preprocessor.preprocess_batch(test_transactions)

    for i, txn in enumerate(preprocessed, 1):
        print(f"\nTransaction {i}:")
        print("-" * 80)
        summary = preprocessor.get_feature_summary(txn)
        for key, value in summary.items():
            print(f"  {key:20s}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    main()
