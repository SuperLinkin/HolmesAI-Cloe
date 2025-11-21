"""
Data ingestion module for loading transaction data from various sources.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Union
from datetime import datetime
from .schema import TransactionInput, TransactionNormalized


class DataIngestion:
    """Handles ingestion and normalization of transaction data from multiple sources."""

    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.xlsx']

    def load_from_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load transaction data from file (CSV, JSON, or Excel).

        Args:
            file_path: Path to the transaction data file

        Returns:
            DataFrame with raw transaction data

        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )

        # Load data based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)

        return df

    def validate_schema(self, df: pd.DataFrame) -> List[TransactionInput]:
        """
        Validate and convert DataFrame to TransactionInput objects.

        Args:
            df: DataFrame with transaction data

        Returns:
            List of validated TransactionInput objects
        """
        transactions = []
        errors = []

        for idx, row in df.iterrows():
            try:
                # Convert timestamp to datetime if string
                if isinstance(row.get('timestamp'), str):
                    row['timestamp'] = pd.to_datetime(row['timestamp'])

                # Create TransactionInput object (validates schema)
                transaction = TransactionInput(**row.to_dict())
                transactions.append(transaction)
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")

        if errors:
            print(f"Validation errors found in {len(errors)} transactions:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

        print(f"Successfully validated {len(transactions)} transactions")
        return transactions

    def normalize_transactions(self, transactions: List[TransactionInput]) -> List[TransactionNormalized]:
        """
        Normalize transactions by extracting temporal features.

        Args:
            transactions: List of validated transaction inputs

        Returns:
            List of normalized transactions with derived features
        """
        normalized = []

        for txn in transactions:
            # Extract temporal features
            day_of_week = txn.timestamp.weekday()  # 0=Monday, 6=Sunday
            hour_of_day = txn.timestamp.hour
            month = txn.timestamp.month

            # Create normalized transaction (merchant_cleaned will be set by preprocessing)
            normalized_txn = TransactionNormalized(
                transaction_id=txn.transaction_id,
                merchant_raw=txn.merchant_raw,
                merchant_cleaned=txn.merchant_raw,  # Will be cleaned by preprocessing module
                amount=txn.amount,
                currency=txn.currency,
                timestamp=txn.timestamp,
                channel=txn.channel,
                location=txn.location,
                mcc_code=txn.mcc_code,
                day_of_week=day_of_week,
                hour_of_day=hour_of_day,
                month=month
            )
            normalized.append(normalized_txn)

        return normalized

    def ingest_pipeline(self, file_path: Union[str, Path]) -> List[TransactionNormalized]:
        """
        Complete ingestion pipeline: load -> validate -> normalize.

        Args:
            file_path: Path to transaction data file

        Returns:
            List of normalized transactions ready for preprocessing
        """
        print(f"Starting data ingestion from: {file_path}")

        # Step 1: Load data
        df = self.load_from_file(file_path)
        print(f"Loaded {len(df)} records from file")

        # Step 2: Validate schema
        transactions = self.validate_schema(df)

        # Step 3: Normalize
        normalized = self.normalize_transactions(transactions)
        print(f"Normalized {len(normalized)} transactions")

        return normalized

    def export_normalized(self, transactions: List[TransactionNormalized], output_path: Union[str, Path]):
        """
        Export normalized transactions to file.

        Args:
            transactions: List of normalized transactions
            output_path: Path to save the normalized data
        """
        output_path = Path(output_path)

        # Convert to DataFrame
        data = [txn.model_dump() for txn in transactions]
        df = pd.DataFrame(data)

        # Save based on extension
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            df.to_json(output_path, orient='records', date_format='iso')
        elif output_path.suffix == '.xlsx':
            df.to_excel(output_path, index=False)

        print(f"Exported {len(transactions)} normalized transactions to: {output_path}")


def main():
    """Example usage of the DataIngestion module."""

    # Initialize ingestion
    ingestion = DataIngestion()

    # Example: Ingest from CSV
    # normalized_transactions = ingestion.ingest_pipeline('data/raw/transactions.csv')

    # Example: Export normalized data
    # ingestion.export_normalized(normalized_transactions, 'data/processed/transactions_normalized.csv')

    print("DataIngestion module ready for use")


if __name__ == "__main__":
    main()
