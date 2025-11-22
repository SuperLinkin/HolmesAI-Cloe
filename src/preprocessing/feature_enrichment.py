"""
Feature enrichment module for transaction data.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class SpendBand(Enum):
    """Spending bands for amount categorization."""
    MICRO = "micro"      # < $10
    SMALL = "small"      # $10-50
    MEDIUM = "medium"    # $50-200
    LARGE = "large"      # > $200


class TemporalPattern(Enum):
    """Temporal patterns for transaction timing."""
    WEEKDAY_MORNING = "weekday_morning"      # Mon-Fri, 6-9 AM
    WEEKDAY_LUNCH = "weekday_lunch"          # Mon-Fri, 11 AM-2 PM
    WEEKDAY_EVENING = "weekday_evening"      # Mon-Fri, 5-9 PM
    WEEKDAY_NIGHT = "weekday_night"          # Mon-Fri, 9 PM-6 AM
    WEEKEND_DAY = "weekend_day"              # Sat-Sun, 6 AM-9 PM
    WEEKEND_NIGHT = "weekend_night"          # Sat-Sun, 9 PM-6 AM


class FeatureEnrichment:
    """Enrich transaction data with derived features."""

    def __init__(self):
        # Spend band thresholds (USD)
        self.spend_thresholds = {
            SpendBand.MICRO: (0, 10),
            SpendBand.SMALL: (10, 50),
            SpendBand.MEDIUM: (50, 200),
            SpendBand.LARGE: (200, float('inf'))
        }

    def get_spend_band(self, amount: float, currency: str = "USD") -> str:
        """
        Categorize transaction amount into spending bands.

        Args:
            amount: Transaction amount
            currency: Currency code (for future multi-currency support)

        Returns:
            Spend band category
        """
        # Convert to absolute value (handle refunds)
        abs_amount = abs(amount)

        # TODO: Add currency conversion for non-USD amounts
        # For now, assume USD or treat as equivalent

        for band, (min_val, max_val) in self.spend_thresholds.items():
            if min_val <= abs_amount < max_val:
                return band.value

        return SpendBand.LARGE.value

    def get_temporal_pattern(self, day_of_week: int, hour_of_day: int) -> str:
        """
        Identify temporal pattern based on day and time.

        Args:
            day_of_week: Day of week (0=Monday, 6=Sunday)
            hour_of_day: Hour of day (0-23)

        Returns:
            Temporal pattern category
        """
        is_weekend = day_of_week >= 5  # Saturday or Sunday

        if is_weekend:
            if 6 <= hour_of_day < 21:
                return TemporalPattern.WEEKEND_DAY.value
            else:
                return TemporalPattern.WEEKEND_NIGHT.value
        else:  # Weekday
            if 6 <= hour_of_day < 9:
                return TemporalPattern.WEEKDAY_MORNING.value
            elif 11 <= hour_of_day < 14:
                return TemporalPattern.WEEKDAY_LUNCH.value
            elif 17 <= hour_of_day < 21:
                return TemporalPattern.WEEKDAY_EVENING.value
            else:
                return TemporalPattern.WEEKDAY_NIGHT.value

    def extract_geographic_features(self, location: Optional[str]) -> Dict[str, str]:
        """
        Extract geographic features from location string.

        Args:
            location: Location string (e.g., "Seattle, WA")

        Returns:
            Dictionary with city and region
        """
        if not location:
            return {"city": "unknown", "region": "unknown"}

        # Simple parsing - can be enhanced with geopy or similar
        parts = [p.strip() for p in location.split(',')]

        city = parts[0] if len(parts) > 0 else "unknown"
        region = parts[-1] if len(parts) > 1 else "unknown"

        return {
            "city": city.lower(),
            "region": region.lower()
        }

    def get_mcc_category_group(self, mcc_code: Optional[str]) -> str:
        """
        Map MCC code to broad category groups.

        Args:
            mcc_code: 4-digit Merchant Category Code

        Returns:
            Category group name
        """
        if not mcc_code:
            return "unknown"

        try:
            mcc = int(mcc_code)
        except (ValueError, TypeError):
            return "unknown"

        # MCC code ranges (simplified mapping)
        if 4000 <= mcc <= 4799:
            return "transportation"
        elif 5000 <= mcc <= 5599:
            return "retail"
        elif 5600 <= mcc <= 5699:
            return "clothing"
        elif 5700 <= mcc <= 5799:
            return "home_garden"
        elif 5800 <= mcc <= 5999:
            return "dining"
        elif 7000 <= mcc <= 7999:
            return "services"
        elif 8000 <= mcc <= 8999:
            return "professional"
        else:
            return "other"

    def enrich_transaction(self, transaction: Dict) -> Dict:
        """
        Enrich a single transaction with derived features.

        Args:
            transaction: Transaction dictionary with base fields

        Returns:
            Dictionary with enriched features
        """
        enriched = transaction.copy()

        # Spend band
        enriched['spend_band'] = self.get_spend_band(
            transaction['amount'],
            transaction.get('currency', 'USD')
        )

        # Extract temporal features from timestamp if not already present
        if 'timestamp' in transaction and ('day_of_week' not in transaction or 'hour_of_day' not in transaction):
            try:
                ts = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
                day_of_week = ts.weekday()
                hour_of_day = ts.hour
            except Exception:
                # Default to weekday afternoon if parsing fails
                day_of_week = 0
                hour_of_day = 14
        else:
            day_of_week = transaction.get('day_of_week', 0)
            hour_of_day = transaction.get('hour_of_day', 14)

        enriched['day_of_week'] = day_of_week
        enriched['hour_of_day'] = hour_of_day

        # Temporal pattern
        enriched['temporal_pattern'] = self.get_temporal_pattern(
            day_of_week,
            hour_of_day
        )

        # Geographic features
        geo_features = self.extract_geographic_features(
            transaction.get('location')
        )
        enriched.update(geo_features)

        # MCC category group
        enriched['mcc_category_group'] = self.get_mcc_category_group(
            transaction.get('mcc_code')
        )

        # Is refund/credit?
        enriched['is_refund'] = transaction['amount'] < 0

        # Day type
        enriched['is_weekend'] = day_of_week >= 5

        return enriched

    def enrich_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Enrich a batch of transactions.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of enriched transactions
        """
        return [self.enrich_transaction(txn) for txn in transactions]


def main():
    """Example usage of FeatureEnrichment."""

    enricher = FeatureEnrichment()

    # Test transaction
    test_txn = {
        'transaction_id': 'TXN_001',
        'amount': 45.50,
        'currency': 'USD',
        'day_of_week': 0,  # Monday
        'hour_of_day': 18,  # 6 PM
        'month': 1,
        'location': 'Seattle, WA',
        'mcc_code': '5814'
    }

    print("Feature Enrichment Example:")
    print("-" * 60)
    print("Original transaction:")
    for key, value in test_txn.items():
        print(f"  {key}: {value}")

    enriched = enricher.enrich_transaction(test_txn)

    print("\nEnriched features:")
    for key, value in enriched.items():
        if key not in test_txn:
            print(f"  {key}: {value}")
    print("-" * 60)


if __name__ == "__main__":
    main()
