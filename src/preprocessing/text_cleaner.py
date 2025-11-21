"""
Text cleaning and normalization for merchant descriptions.
"""

import re
from typing import List, Set


class TextCleaner:
    """Clean and normalize merchant transaction descriptions."""

    def __init__(self):
        # Payment processor prefixes to remove
        self.payment_processors = [
            r'SQ\*',        # Square
            r'PAYPAL\*',    # PayPal
            r'AMZN\s*MKTP', # Amazon Marketplace
            r'GOOGLE\*',    # Google Pay
            r'STRIPE\*',    # Stripe
            r'TST\*',       # Toast
            r'SPT\*',       # Shopify
        ]

        # Special characters to remove
        self.special_chars = r'[*#@$%^&()\[\]{}|\\/<>+=_~`]'

        # Common suffixes to remove
        self.suffixes = [
            r'\d{4,}',      # Long numeric sequences (transaction IDs)
            r'[A-Z0-9]{8,}', # Long alphanumeric codes
            r'\b(STORE|POS|PURCHASE|PAYMENT|TRANSACTION)\b',
        ]

    def remove_payment_processors(self, text: str) -> str:
        """
        Remove payment processor prefixes from merchant names.

        Args:
            text: Raw merchant description

        Returns:
            Text with payment processors removed
        """
        for processor in self.payment_processors:
            text = re.sub(processor, '', text, flags=re.IGNORECASE)
        return text

    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters while preserving spaces and hyphens.

        Args:
            text: Input text

        Returns:
            Text with special characters removed
        """
        # Replace special chars with space
        text = re.sub(self.special_chars, ' ', text)
        return text

    def remove_suffixes(self, text: str) -> str:
        """
        Remove common suffixes like transaction IDs and noise words.

        Args:
            text: Input text

        Returns:
            Text with suffixes removed
        """
        for suffix in self.suffixes:
            text = re.sub(suffix, '', text, flags=re.IGNORECASE)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace: multiple spaces to single space, strip edges.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()

    def extract_tokens(self, text: str) -> List[str]:
        """
        Extract meaningful tokens from cleaned text.

        Args:
            text: Cleaned text

        Returns:
            List of tokens
        """
        # Split on whitespace
        tokens = text.lower().split()

        # Filter out very short tokens (likely noise)
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def clean(self, merchant_raw: str) -> str:
        """
        Complete cleaning pipeline for merchant descriptions.

        Args:
            merchant_raw: Raw merchant description from bank feed

        Returns:
            Cleaned merchant name
        """
        text = merchant_raw

        # Step 1: Remove payment processors
        text = self.remove_payment_processors(text)

        # Step 2: Remove special characters
        text = self.remove_special_characters(text)

        # Step 3: Remove suffixes and noise
        text = self.remove_suffixes(text)

        # Step 4: Normalize whitespace
        text = self.normalize_whitespace(text)

        # Step 5: Convert to lowercase
        text = text.lower()

        return text

    def clean_batch(self, merchants: List[str]) -> List[str]:
        """
        Clean a batch of merchant descriptions.

        Args:
            merchants: List of raw merchant descriptions

        Returns:
            List of cleaned merchant names
        """
        return [self.clean(m) for m in merchants]


def main():
    """Example usage of TextCleaner."""

    cleaner = TextCleaner()

    # Test cases
    test_cases = [
        "AMZN MKTP US*2A3B4C5D6 SEATTLE WA",
        "SQ *COFFEE SHOP #12345",
        "PAYPAL *NETFLIX.COM",
        "SWIGGY*FOOD DELIVERY 98765",
        "UBER *TRIP BANGALORE"
    ]

    print("Text Cleaning Examples:")
    print("-" * 60)

    for raw in test_cases:
        cleaned = cleaner.clean(raw)
        tokens = cleaner.extract_tokens(cleaned)
        print(f"Raw:     {raw}")
        print(f"Cleaned: {cleaned}")
        print(f"Tokens:  {tokens}")
        print("-" * 60)


if __name__ == "__main__":
    main()
