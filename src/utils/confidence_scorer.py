"""
Confidence scoring system for transaction categorization.
"""

import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path


class ConfidenceScorer:
    """
    Calculate confidence scores for transaction categorization predictions.

    Confidence score is computed as:
    Confidence_L3 = (LightGBM_Probability × 0.60) +
                    (Alias_Match_Score × 0.25) +
                    (MCC_Alignment_Score × 0.15)
    """

    def __init__(self, taxonomy_path: Optional[str] = None):
        """
        Initialize confidence scorer.

        Args:
            taxonomy_path: Path to taxonomy.json file
        """
        self.taxonomy = None
        self.mcc_to_l3 = {}
        self.alias_to_l3 = {}

        if taxonomy_path:
            self.load_taxonomy(taxonomy_path)

        # Confidence thresholds
        self.THRESHOLD_HIGH = 0.85
        self.THRESHOLD_MEDIUM = 0.70
        self.THRESHOLD_LOW = 0.70

        # Weights for confidence components
        self.WEIGHT_MODEL = 0.60
        self.WEIGHT_ALIAS = 0.25
        self.WEIGHT_MCC = 0.15

    def load_taxonomy(self, taxonomy_path: str):
        """
        Load taxonomy and build lookup dictionaries.

        Args:
            taxonomy_path: Path to taxonomy.json
        """
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

        # Build MCC and alias lookup tables
        for l1_cat in self.taxonomy['categories']:
            for l2_cat in l1_cat['l2_subcategories']:
                for l3_cat in l2_cat['l3_types']:
                    l3_id = l3_cat['l3_id']

                    # MCC mapping
                    for mcc in l3_cat.get('mcc_codes', []):
                        self.mcc_to_l3[mcc] = l3_id

                    # Alias mapping
                    for alias in l3_cat.get('aliases', []):
                        self.alias_to_l3[alias.lower()] = l3_id

        print(f"Loaded taxonomy with {len(self.mcc_to_l3)} MCC mappings "
              f"and {len(self.alias_to_l3)} alias mappings")

    def compute_model_confidence(
        self,
        probabilities: np.ndarray,
        predicted_class_idx: int
    ) -> float:
        """
        Extract model confidence from probability distribution.

        Args:
            probabilities: Probability distribution over all classes
            predicted_class_idx: Index of predicted class

        Returns:
            Model confidence (0-1)
        """
        return float(probabilities[predicted_class_idx])

    def compute_alias_match_score(
        self,
        merchant_cleaned: str,
        merchant_tokens: List[str],
        predicted_l3_id: str,
        similarity_score: Optional[float] = None
    ) -> float:
        """
        Compute alias match score.

        Args:
            merchant_cleaned: Cleaned merchant name
            merchant_tokens: Tokenized merchant name
            predicted_l3_id: Predicted L3 category ID
            similarity_score: Optional cosine similarity to known merchant

        Returns:
            Alias match score (0-1)
        """
        # Check if merchant matches known alias
        merchant_lower = merchant_cleaned.lower()

        # Get aliases for predicted category
        predicted_aliases = []
        for l1_cat in self.taxonomy['categories']:
            for l2_cat in l1_cat['l2_subcategories']:
                for l3_cat in l2_cat['l3_types']:
                    if l3_cat['l3_id'] == predicted_l3_id:
                        predicted_aliases = [a.lower() for a in l3_cat.get('aliases', [])]
                        break

        # Exact alias match
        if merchant_lower in predicted_aliases:
            return 1.0

        # Token overlap with aliases
        for alias in predicted_aliases:
            alias_tokens = alias.split()
            overlap = len(set(merchant_tokens) & set(alias_tokens))
            if overlap > 0 and len(alias_tokens) > 0:
                overlap_ratio = overlap / len(alias_tokens)
                if overlap_ratio >= 0.7:
                    return 0.9

        # Use similarity score if available
        if similarity_score is not None:
            if similarity_score >= 0.90:
                return 0.9
            elif similarity_score >= 0.80:
                return 0.7
            elif similarity_score >= 0.70:
                return 0.5

        return 0.0

    def compute_mcc_alignment_score(
        self,
        mcc_code: Optional[str],
        predicted_l3_id: str,
        predicted_l2_id: str
    ) -> float:
        """
        Compute MCC alignment score.

        Args:
            mcc_code: Transaction MCC code
            predicted_l3_id: Predicted L3 category ID
            predicted_l2_id: Predicted L2 category ID

        Returns:
            MCC alignment score (0-1)
        """
        if not mcc_code:
            return 0.0

        # Check if MCC matches predicted L3
        if mcc_code in self.mcc_to_l3:
            if self.mcc_to_l3[mcc_code] == predicted_l3_id:
                return 1.0

        # Check if MCC matches predicted L2 (partial match)
        # Get L2 for the MCC's L3
        mcc_l3_id = self.mcc_to_l3.get(mcc_code)
        if mcc_l3_id:
            # Extract L2 from MCC's L3
            mcc_l2_id = '-'.join(mcc_l3_id.split('-')[:2])
            pred_l2_id = '-'.join(predicted_l2_id.split('-')[:2])

            if mcc_l2_id == pred_l2_id:
                return 0.5

        return 0.0

    def compute_confidence(
        self,
        model_probability: float,
        alias_match_score: float,
        mcc_alignment_score: float
    ) -> float:
        """
        Compute overall confidence score.

        Args:
            model_probability: Model confidence (0-1)
            alias_match_score: Alias match score (0-1)
            mcc_alignment_score: MCC alignment score (0-1)

        Returns:
            Overall confidence score (0-1)
        """
        confidence = (
            self.WEIGHT_MODEL * model_probability +
            self.WEIGHT_ALIAS * alias_match_score +
            self.WEIGHT_MCC * mcc_alignment_score
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level category.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        if confidence >= self.THRESHOLD_HIGH:
            return 'high'
        elif confidence >= self.THRESHOLD_MEDIUM:
            return 'medium'
        else:
            return 'low'

    def should_review(self, confidence: float) -> bool:
        """
        Determine if transaction should be flagged for manual review.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            True if should be reviewed, False otherwise
        """
        return confidence < self.THRESHOLD_LOW

    def score_prediction(
        self,
        transaction: Dict,
        l3_probabilities: np.ndarray,
        predicted_l3_idx: int,
        predicted_l3_id: str,
        predicted_l2_id: str,
        similarity_score: Optional[float] = None
    ) -> Dict:
        """
        Score a prediction with all components.

        Args:
            transaction: Transaction dictionary
            l3_probabilities: L3 probability distribution
            predicted_l3_idx: Index of predicted L3 class
            predicted_l3_id: Predicted L3 category ID
            predicted_l2_id: Predicted L2 category ID
            similarity_score: Optional similarity to known merchant

        Returns:
            Dictionary with confidence scores and components
        """
        # Model confidence
        model_conf = self.compute_model_confidence(l3_probabilities, predicted_l3_idx)

        # Alias match score
        alias_score = self.compute_alias_match_score(
            transaction.get('merchant_cleaned', ''),
            transaction.get('merchant_tokens', []),
            predicted_l3_id,
            similarity_score
        )

        # MCC alignment score
        mcc_score = self.compute_mcc_alignment_score(
            transaction.get('mcc_code'),
            predicted_l3_id,
            predicted_l2_id
        )

        # Overall confidence
        confidence = self.compute_confidence(model_conf, alias_score, mcc_score)

        return {
            'confidence': confidence,
            'confidence_level': self.get_confidence_level(confidence),
            'should_review': self.should_review(confidence),
            'components': {
                'model_probability': model_conf,
                'alias_match_score': alias_score,
                'mcc_alignment_score': mcc_score
            },
            'weights': {
                'model': self.WEIGHT_MODEL,
                'alias': self.WEIGHT_ALIAS,
                'mcc': self.WEIGHT_MCC
            }
        }


def main():
    """Example usage of ConfidenceScorer."""

    # Initialize scorer (without taxonomy for demo)
    scorer = ConfidenceScorer()

    # Example prediction
    l3_probs = np.array([0.05, 0.10, 0.75, 0.08, 0.02])  # 45 classes total
    predicted_idx = 2

    # Example transaction
    transaction = {
        'merchant_cleaned': 'amazon marketplace',
        'merchant_tokens': ['amazon', 'marketplace'],
        'mcc_code': '5942'
    }

    print("Confidence Scoring Example:")
    print("=" * 80)

    # Compute individual scores
    model_conf = scorer.compute_model_confidence(l3_probs, predicted_idx)
    print(f"Model Confidence: {model_conf:.4f}")

    # Overall confidence (without taxonomy)
    confidence = scorer.compute_confidence(
        model_probability=model_conf,
        alias_match_score=0.8,  # Example
        mcc_alignment_score=1.0  # Example
    )

    print(f"Overall Confidence: {confidence:.4f}")
    print(f"Confidence Level: {scorer.get_confidence_level(confidence)}")
    print(f"Should Review: {scorer.should_review(confidence)}")


if __name__ == "__main__":
    main()
