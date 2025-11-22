"""
Synthetic Dataset Generator for Holmes AI Transaction Categorization Engine

This script generates a realistic synthetic dataset for training the transaction
categorization model using the taxonomy structure and MCC codes from Visa/Mastercard/Amex.
"""

import json
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np


class SyntheticDatasetGenerator:
    """Generates synthetic transaction data based on the taxonomy."""

    def __init__(self, taxonomy_path: str = "src/config/taxonomy.json"):
        """Initialize the generator with taxonomy configuration."""
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

        self.merchant_templates = self._load_merchant_templates()
        self.channels = ["online", "pos", "atm", "mobile"]
        self.currencies = ["USD", "EUR", "GBP", "INR", "AUD"]

    def _load_merchant_templates(self) -> Dict[str, List[Dict]]:
        """Extract merchant templates from taxonomy."""
        templates = {}

        for l1_cat in self.taxonomy['categories']:
            for l2_cat in l1_cat['l2_subcategories']:
                for l3_cat in l2_cat['l3_types']:
                    l3_id = l3_cat['l3_id']
                    templates[l3_id] = {
                        'l1': l1_cat['l1'],
                        'l1_id': l1_cat['l1_id'],
                        'l2': l2_cat['l2'],
                        'l2_id': l2_cat['l2_id'],
                        'l3': l3_cat['l3'],
                        'l3_id': l3_id,
                        'aliases': l3_cat['aliases'],
                        'mcc_codes': l3_cat['mcc_codes'],
                        'keywords': l3_cat['keywords']
                    }

        return templates

    def _generate_merchant_name(self, template: Dict) -> str:
        """Generate a realistic merchant name with variations."""
        base_alias = random.choice(template['aliases'])

        # Patterns for realistic merchant names
        patterns = [
            "{alias}",
            "{alias} {location}",
            "{alias} #{number}",
            "{alias} {location} #{number}",
            "{prefix} {alias}",
            "{alias} {suffix}",
            "{alias} * {code}",
            "{payment_ref} {alias}",
        ]

        locations = ["NY", "CA", "TX", "FL", "WA", "IL", "MA", "GA", "NC", "VA",
                     "MUMBAI", "DELHI", "BANGALORE", "PUNE", "HYDERABAD"]

        prefixes = ["SQ *", "TST*", "PAYPAL *", "STRIPE *", ""]
        suffixes = ["INC", "LLC", "LTD", "STORE", "ONLINE", ""]

        pattern = random.choice(patterns)
        merchant_name = pattern.format(
            alias=base_alias.upper(),
            location=random.choice(locations),
            number=random.randint(100, 9999),
            prefix=random.choice(prefixes),
            suffix=random.choice(suffixes),
            code=''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8)),
            payment_ref=random.choice(["SQ *", "TST*", "PP*", ""])
        )

        # Add random noise to simulate real merchant names
        if random.random() < 0.3:
            merchant_name += " " + ''.join(random.choices('0123456789', k=random.randint(3, 6)))

        return merchant_name.strip()

    def _generate_amount(self, l1_category: str) -> float:
        """Generate realistic transaction amounts based on category."""
        amount_ranges = {
            "Travel": (10, 2000),
            "Dining": (5, 150),
            "Shopping": (10, 500),
            "Entertainment": (5, 200),
            "Utilities": (20, 300),
            "Healthcare": (15, 500),
            "Education": (50, 5000),
            "Transportation": (5, 100),
            "Financial Services": (10, 10000),
            "Housing": (500, 3000),
            "Personal Care": (10, 200),
            "Technology": (20, 2000),
            "Charitable": (5, 500),
            "Business Expenses": (20, 1000),
            "Miscellaneous": (5, 200)
        }

        min_amt, max_amt = amount_ranges.get(l1_category, (5, 500))

        # Use log-normal distribution for more realistic amounts
        mu = np.log(min_amt + (max_amt - min_amt) / 3)
        sigma = 0.8
        amount = np.random.lognormal(mu, sigma)
        amount = max(min_amt, min(max_amt, amount))

        return round(amount, 2)

    def _generate_timestamp(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate realistic timestamp with temporal patterns."""
        # More transactions on weekdays, during business hours
        timestamp = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )

        # Adjust for realistic patterns
        if timestamp.weekday() < 5:  # Weekday
            if random.random() < 0.7:  # 70% during business hours
                timestamp = timestamp.replace(hour=random.randint(9, 20))

        return timestamp

    def _select_channel(self, l3_id: str) -> str:
        """Select appropriate payment channel based on category."""
        online_categories = ["SHP-ONL", "ENT-STR", "TEC-SUB", "EDU-CRS"]
        pos_categories = ["SHP-RET", "DIN-RES", "DIN-FST", "TRN-FUL"]

        if any(cat in l3_id for cat in online_categories):
            return "online" if random.random() < 0.9 else random.choice(self.channels)
        elif any(cat in l3_id for cat in pos_categories):
            return "pos" if random.random() < 0.8 else random.choice(self.channels)

        return random.choice(self.channels)

    def _generate_location(self, channel: str) -> str:
        """Generate location based on channel."""
        if channel == "online":
            return None if random.random() < 0.7 else random.choice([
                "Online", "Internet", "Web Purchase"
            ])

        cities = [
            "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
            "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
            "Mumbai, MH", "Delhi, DL", "Bangalore, KA", "Hyderabad, TG",
            "Chennai, TN", "Kolkata, WB", "Pune, MH", "Ahmedabad, GJ"
        ]

        return random.choice(cities)

    def generate_transaction(
        self,
        transaction_id: str,
        template: Dict,
        timestamp: datetime,
        currency: str = "USD"
    ) -> Dict:
        """Generate a single synthetic transaction."""
        merchant_name = self._generate_merchant_name(template)
        amount = self._generate_amount(template['l1'])
        channel = self._select_channel(template['l3_id'])
        location = self._generate_location(channel)
        mcc_code = random.choice(template['mcc_codes']) if template['mcc_codes'] else None

        transaction = {
            'transaction_id': transaction_id,
            'merchant_raw': merchant_name,
            'amount': amount,
            'currency': currency,
            'timestamp': timestamp.isoformat(),
            'channel': channel,
            'location': location,
            'mcc_code': mcc_code,
            # Ground truth labels
            'l1': template['l1'],
            'l1_id': template['l1_id'],
            'l2': template['l2'],
            'l2_id': template['l2_id'],
            'l3': template['l3'],
            'l3_id': template['l3_id']
        }

        return transaction

    def generate_dataset(
        self,
        num_samples: int = 10000,
        start_date: datetime = None,
        end_date: datetime = None,
        output_path: str = "data/synthetic_transactions.csv"
    ) -> pd.DataFrame:
        """Generate a complete synthetic dataset."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        print(f"Generating {num_samples} synthetic transactions...")

        transactions = []

        # Calculate category distribution (some categories more common than others)
        category_weights = self._calculate_category_weights()

        for i in range(num_samples):
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_samples} transactions...")

            # Select category based on weights
            l3_id = random.choices(
                list(self.merchant_templates.keys()),
                weights=category_weights,
                k=1
            )[0]

            template = self.merchant_templates[l3_id]
            transaction_id = f"TXN_{start_date.strftime('%Y%m%d')}_{i:06d}"
            timestamp = self._generate_timestamp(start_date, end_date)
            currency = random.choices(
                self.currencies,
                weights=[0.5, 0.2, 0.1, 0.15, 0.05],  # USD most common
                k=1
            )[0]

            transaction = self.generate_transaction(
                transaction_id, template, timestamp, currency
            )
            transactions.append(transaction)

        df = pd.DataFrame(transactions)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to {output_path}")
        print(f"Total transactions: {len(df)}")

        # Print statistics
        self._print_statistics(df)

        return df

    def _calculate_category_weights(self) -> List[float]:
        """Calculate realistic distribution weights for categories."""
        # Some categories are more common in real life
        l3_ids = list(self.merchant_templates.keys())
        weights = []

        for l3_id in l3_ids:
            template = self.merchant_templates[l3_id]
            l1 = template['l1']

            # Higher frequency categories
            if l1 in ["Dining", "Shopping", "Transportation"]:
                weight = 3.0
            elif l1 in ["Entertainment", "Utilities"]:
                weight = 2.0
            elif l1 in ["Travel", "Healthcare", "Personal Care"]:
                weight = 1.5
            else:
                weight = 1.0

            weights.append(weight)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        return weights

    def _print_statistics(self, df: pd.DataFrame):
        """Print dataset statistics."""
        print("\n=== Dataset Statistics ===")
        print(f"\nL1 Category Distribution:")
        print(df['l1'].value_counts())

        print(f"\nCurrency Distribution:")
        print(df['currency'].value_counts())

        print(f"\nChannel Distribution:")
        print(df['channel'].value_counts())

        print(f"\nAmount Statistics:")
        print(df['amount'].describe())

        print(f"\nDate Range:")
        print(f"From: {df['timestamp'].min()}")
        print(f"To: {df['timestamp'].max()}")

        print(f"\nL2 Categories: {df['l2'].nunique()}")
        print(f"L3 Categories: {df['l3'].nunique()}")

    def generate_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        output_dir: str = "data"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split

        # Stratified split based on L3 categories
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df['l3_id'],
            random_state=42
        )

        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            stratify=train_val['l3_id'],
            random_state=42
        )

        # Save splits
        train.to_csv(f"{output_dir}/train.csv", index=False)
        val.to_csv(f"{output_dir}/val.csv", index=False)
        test.to_csv(f"{output_dir}/test.csv", index=False)

        print(f"\nDataset split:")
        print(f"Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
        print(f"Val: {len(val)} ({len(val)/len(df)*100:.1f}%)")
        print(f"Test: {len(test)} ({len(test)/len(df)*100:.1f}%)")

        return train, val, test


def main():
    """Main execution function."""
    import os

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Initialize generator
    generator = SyntheticDatasetGenerator()

    # Generate dataset
    print("=" * 60)
    print("Holmes AI - Synthetic Dataset Generator")
    print("=" * 60)

    # Generate different sized datasets
    datasets = [
        (1000, "data/synthetic_transactions_1k.csv"),
        (10000, "data/synthetic_transactions_10k.csv"),
        (50000, "data/synthetic_transactions_50k.csv"),
    ]

    for num_samples, output_path in datasets:
        print(f"\n{'=' * 60}")
        df = generator.generate_dataset(
            num_samples=num_samples,
            output_path=output_path
        )

        # Create train/val/test splits for the largest dataset
        if num_samples == 50000:
            print(f"\n{'=' * 60}")
            print("Creating train/val/test splits...")
            generator.generate_train_test_split(df)

    print(f"\n{'=' * 60}")
    print("✓ Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
