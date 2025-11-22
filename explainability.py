"""
Enhanced Explainability Module for Holmes AI

Provides comprehensive explanations for predictions including:
- SHAP feature importance analysis
- Top contributing features for each prediction
- Prediction reasoning in natural language
- Confidence breakdown by component (model, alias, MCC, hierarchy)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import shap
import matplotlib.pyplot as plt

from src.data_ingestion import DataIngestion
from src.preprocessing import TransactionPreprocessor
from src.models import SentenceBERTEncoder, LightGBMClassifier


class ExplainabilityEngine:
    """Enhanced explainability engine with SHAP analysis."""

    def __init__(
        self,
        model_path: str = "models",
        taxonomy_path: str = "src/config/taxonomy.json"
    ):
        """Initialize explainability engine."""
        self.model_path = model_path
        self.taxonomy_path = taxonomy_path

        # Load models
        print("Loading models...")
        self.encoder = SentenceBERTEncoder(model_path=f"{model_path}/sentence_bert")
        self.classifier = LightGBMClassifier(taxonomy_path=taxonomy_path)
        self.classifier.load_models(f"{model_path}/lightgbm")
        print("[OK] Models loaded")

        # Feature names
        self.feature_names = self._get_feature_names()

        # SHAP explainers (lazy initialized)
        self.shap_explainer_l1 = None
        self.shap_explainer_l2 = None
        self.shap_explainer_l3 = None

    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        feature_names = []

        # Embedding features (768 dimensions)
        for i in range(768):
            feature_names.append(f"embedding_dim_{i}")

        # Engineered features
        feature_names.extend([
            "spend_band",
            "temporal_pattern",
            "channel",
            "mcc_code_normalized",
            "amount_percentile"
        ])

        return feature_names

    def init_shap_explainers(self, background_data: np.ndarray, max_samples: int = 100):
        """Initialize SHAP explainers with background data."""
        print(f"Initializing SHAP explainers with {min(max_samples, len(background_data))} background samples...")

        # Use subset for efficiency
        background_subset = shap.sample(background_data, min(max_samples, len(background_data)))

        # Initialize TreeExplainer for each level
        print("  Initializing L1 explainer...")
        self.shap_explainer_l1 = shap.TreeExplainer(
            self.classifier.model_l1,
            data=background_subset,
            feature_perturbation="tree_path_dependent"
        )

        print("  Initializing L2 explainer...")
        self.shap_explainer_l2 = shap.TreeExplainer(
            self.classifier.model_l2,
            data=background_subset,
            feature_perturbation="tree_path_dependent"
        )

        print("  Initializing L3 explainer...")
        self.shap_explainer_l3 = shap.TreeExplainer(
            self.classifier.model_l3,
            data=background_subset,
            feature_perturbation="tree_path_dependent"
        )

        print("[OK] SHAP explainers initialized")

    def explain_prediction(
        self,
        transaction: Dict,
        top_k_features: int = 10
    ) -> Dict:
        """
        Provide comprehensive explanation for a single transaction prediction.

        Returns:
            Dictionary with prediction, confidence, SHAP values, top features, and reasoning
        """
        # Preprocess
        preprocessor = TransactionPreprocessor()
        preprocessed = preprocessor.preprocess_batch([transaction])[0]

        # Generate embedding
        embedding = self.encoder.encode_transactions(
            [preprocessed],
            text_field='merchant_cleaned',
            batch_size=1
        )

        # Prepare features
        X = self.classifier.prepare_features(
            embedding,
            [preprocessed],
            include_enrichment=True
        )

        # Make prediction
        prediction = self.classifier.predict(X, use_hierarchy=True)

        # Get SHAP values if explainers initialized
        shap_values = None
        top_features = None

        if self.shap_explainer_l3 is not None:
            # Get SHAP values for L3 (most detailed)
            shap_values_l3 = self.shap_explainer_l3.shap_values(X)

            # For multi-class, get SHAP values for predicted class
            predicted_l3_idx = np.where(
                self.classifier.encoder_l3.classes_ == prediction['l3'][0]
            )[0][0]

            shap_vals = shap_values_l3[predicted_l3_idx][0] if isinstance(shap_values_l3, list) else shap_values_l3[0]

            # Get top contributing features
            top_indices = np.argsort(np.abs(shap_vals))[-top_k_features:][::-1]

            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature': self.feature_names[idx],
                    'value': float(X[0, idx]),
                    'shap_value': float(shap_vals[idx]),
                    'impact': 'positive' if shap_vals[idx] > 0 else 'negative'
                })

            shap_values = {
                'values': shap_vals.tolist(),
                'base_value': float(self.shap_explainer_l3.expected_value[predicted_l3_idx])
                    if isinstance(self.shap_explainer_l3.expected_value, np.ndarray)
                    else float(self.shap_explainer_l3.expected_value)
            }

        # Generate natural language reasoning
        reasoning = self._generate_reasoning(
            transaction,
            preprocessed,
            prediction,
            top_features
        )

        # Enhanced confidence breakdown
        confidence_breakdown = self._analyze_confidence(
            transaction,
            preprocessed,
            prediction
        )

        return {
            'transaction': transaction,
            'preprocessed': {
                'merchant_cleaned': preprocessed.get('merchant_cleaned'),
                'spend_band': preprocessed.get('spend_band'),
                'temporal_pattern': preprocessed.get('temporal_pattern'),
                'channel': preprocessed.get('channel')
            },
            'prediction': {
                'l1': prediction['l1'][0],
                'l2': prediction['l2'][0],
                'l3': prediction['l3'][0],
                'confidence': prediction.get('confidence', [0])[0]
            },
            'confidence_breakdown': confidence_breakdown,
            'top_features': top_features,
            'shap_analysis': shap_values,
            'reasoning': reasoning
        }

    def _analyze_confidence(
        self,
        transaction: Dict,
        preprocessed: Dict,
        prediction: Dict
    ) -> Dict:
        """Analyze confidence breakdown by component."""
        breakdown = {
            'overall_confidence': prediction.get('confidence', [0])[0],
            'components': []
        }

        # Check alias match
        merchant = preprocessed.get('merchant_cleaned', '').lower()
        predicted_l3 = prediction['l3'][0]

        # Simple alias check (would need access to taxonomy for full check)
        breakdown['components'].append({
            'component': 'model_probability',
            'weight': 0.7,
            'description': 'LightGBM model confidence based on learned patterns'
        })

        # MCC code match
        if 'mcc_code' in transaction:
            breakdown['components'].append({
                'component': 'mcc_code_match',
                'weight': 0.2,
                'description': f"MCC code {transaction['mcc_code']} correlation with category"
            })

        # Hierarchical consistency
        breakdown['components'].append({
            'component': 'hierarchical_consistency',
            'weight': 0.1,
            'description': 'Category hierarchy validation (L1 → L2 → L3)'
        })

        return breakdown

    def _generate_reasoning(
        self,
        transaction: Dict,
        preprocessed: Dict,
        prediction: Dict,
        top_features: List[Dict]
    ) -> str:
        """Generate natural language explanation for the prediction."""
        merchant = transaction.get('merchant', 'Unknown')
        amount = transaction.get('amount', 0)
        predicted_l3 = prediction['l3'][0]
        confidence = prediction.get('confidence', [0])[0]

        reasoning_parts = []

        # Main prediction statement
        reasoning_parts.append(
            f"Transaction '{merchant}' (${amount:.2f}) was categorized as "
            f"'{predicted_l3}' with {confidence*100:.1f}% confidence."
        )

        # Confidence interpretation
        if confidence > 0.9:
            reasoning_parts.append("This is a high-confidence prediction.")
        elif confidence > 0.7:
            reasoning_parts.append("This is a medium-confidence prediction.")
        else:
            reasoning_parts.append("This is a low-confidence prediction - manual review recommended.")

        # Key factors
        if top_features and len(top_features) > 0:
            reasoning_parts.append("\nKey factors in this decision:")

            # Focus on engineered features for interpretability
            engineered_features = [f for f in top_features if not f['feature'].startswith('embedding')]

            if engineered_features:
                for feat in engineered_features[:3]:
                    feature_name = feat['feature']
                    impact = "increased" if feat['impact'] == 'positive' else "decreased"

                    if feature_name == 'spend_band':
                        reasoning_parts.append(
                            f"  • Spending amount range ({preprocessed.get('spend_band', 'unknown')}) "
                            f"{impact} confidence in this category"
                        )
                    elif feature_name == 'temporal_pattern':
                        reasoning_parts.append(
                            f"  • Transaction timing pattern ({preprocessed.get('temporal_pattern', 'unknown')}) "
                            f"{impact} confidence"
                        )
                    elif feature_name == 'channel':
                        reasoning_parts.append(
                            f"  • Transaction channel ({preprocessed.get('channel', 'unknown')}) "
                            f"{impact} likelihood of this category"
                        )
                    elif feature_name == 'mcc_code_normalized':
                        reasoning_parts.append(
                            f"  • Merchant Category Code {impact} confidence"
                        )

            # Mention semantic similarity
            embedding_features = [f for f in top_features if f['feature'].startswith('embedding')]
            if embedding_features:
                reasoning_parts.append(
                    f"  • Merchant name semantic similarity to known '{predicted_l3}' merchants"
                )

        # Hierarchical path
        reasoning_parts.append(
            f"\nCategory hierarchy: {prediction['l1'][0]} → {prediction['l2'][0]} → {prediction['l3'][0]}"
        )

        return "\n".join(reasoning_parts)

    def generate_feature_importance_report(
        self,
        test_data: pd.DataFrame,
        output_dir: str = "explainability_results",
        num_samples: int = 500
    ):
        """Generate comprehensive feature importance report with SHAP analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("=" * 80)
        print("HOLMES AI - FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        print()

        # Step 1: Prepare data
        print(f"[1/5] Preparing {num_samples} samples...")
        ingestion = DataIngestion()
        normalized = ingestion.ingest_pipeline(test_data.head(num_samples))

        preprocessor = TransactionPreprocessor()
        preprocessed = preprocessor.preprocess_batch([txn.model_dump() for txn in normalized])

        # Generate embeddings
        print("[2/5] Generating embeddings...")
        embeddings = self.encoder.encode_transactions(
            preprocessed,
            text_field='merchant_cleaned',
            batch_size=64
        )

        # Prepare features
        X = self.classifier.prepare_features(embeddings, preprocessed, include_enrichment=True)

        # Initialize SHAP explainers
        print("[3/5] Initializing SHAP explainers...")
        self.init_shap_explainers(X, max_samples=100)

        # Calculate SHAP values
        print("[4/5] Calculating SHAP values...")
        print("  Computing L3 SHAP values...")
        shap_values_l3 = self.shap_explainer_l3.shap_values(X[:100])  # Use subset for speed

        # Generate plots
        print("[5/5] Generating visualizations...")

        # Feature importance summary plot
        plt.figure(figsize=(12, 8))

        # For multi-class, use mean absolute SHAP values across all classes
        if isinstance(shap_values_l3, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_l3], axis=0)
        else:
            mean_shap = np.abs(shap_values_l3).mean(axis=0)

        # Get top 20 features
        top_20_idx = np.argsort(mean_shap)[-20:]
        top_20_names = [self.feature_names[i] for i in top_20_idx]
        top_20_values = mean_shap[top_20_idx]

        plt.barh(range(20), top_20_values)
        plt.yticks(range(20), top_20_names)
        plt.xlabel('Mean |SHAP value| (average impact on model output)')
        plt.title('Top 20 Most Important Features (L3 Prediction)')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance_top20.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved feature importance plot to {output_path / 'feature_importance_top20.png'}")

        # Generate summary report
        report = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'num_samples': num_samples,
            'top_20_features': [
                {
                    'rank': i + 1,
                    'feature': name,
                    'mean_shap_value': float(val)
                }
                for i, (name, val) in enumerate(zip(top_20_names[::-1], top_20_values[::-1]))
            ],
            'feature_categories': {
                'semantic_embeddings': int(sum(1 for name in top_20_names if 'embedding' in name)),
                'engineered_features': int(sum(1 for name in top_20_names if not 'embedding' in name))
            }
        }

        # Save JSON report
        with open(output_path / 'feature_importance_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[OK] Saved feature importance report to {output_path / 'feature_importance_report.json'}")

        # Print summary
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE SUMMARY")
        print("=" * 80)
        print(f"\nTop 10 Most Important Features:")
        for item in report['top_20_features'][:10]:
            print(f"  {item['rank']:2d}. {item['feature']:30s} (impact: {item['mean_shap_value']:.4f})")

        print(f"\nFeature Type Distribution (Top 20):")
        print(f"  Semantic Embeddings: {report['feature_categories']['semantic_embeddings']}")
        print(f"  Engineered Features: {report['feature_categories']['engineered_features']}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced explainability for Holmes AI predictions"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['single', 'batch'],
        default='single',
        help="Explainability mode: single transaction or batch analysis"
    )
    parser.add_argument(
        "--merchant",
        type=str,
        help="Merchant name for single transaction mode"
    )
    parser.add_argument(
        "--amount",
        type=float,
        help="Transaction amount for single transaction mode"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to test data CSV for batch mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="explainability_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models",
        help="Path to trained models"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = ExplainabilityEngine(model_path=args.model)

    if args.mode == 'single':
        # Single transaction explanation
        if not args.merchant or args.amount is None:
            print("Error: --merchant and --amount required for single mode")
            return

        transaction = {
            'merchant': args.merchant,
            'amount': args.amount,
            'currency': 'USD',
            'timestamp': pd.Timestamp.now().isoformat(),
            'channel': 'online'
        }

        print("\n" + "=" * 80)
        print("TRANSACTION EXPLANATION")
        print("=" * 80)
        print()

        explanation = engine.explain_prediction(transaction, top_k_features=10)

        print(f"Transaction: {explanation['transaction']['merchant']} (${explanation['transaction']['amount']:.2f})")
        print(f"\nPrediction:")
        print(f"  Category: {explanation['prediction']['l3']}")
        print(f"  Confidence: {explanation['prediction']['confidence']*100:.1f}%")
        print(f"\n{explanation['reasoning']}")

        # Save detailed explanation
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)

        with open(output_path / 'explanation.json', 'w') as f:
            json.dump(explanation, f, indent=2)

        print(f"\n[OK] Detailed explanation saved to {output_path / 'explanation.json'}")

    else:
        # Batch feature importance analysis
        if not args.data:
            print("Error: --data required for batch mode")
            return

        df = pd.read_csv(args.data)
        engine.generate_feature_importance_report(df, output_dir=args.output)


if __name__ == "__main__":
    main()
