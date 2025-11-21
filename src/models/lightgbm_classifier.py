"""
LightGBM classifier for transaction categorization.
"""

import lightgbm as lgb
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json


class LightGBMClassifier:
    """
    LightGBM-based classifier for hierarchical transaction categorization.
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        taxonomy_path: Optional[str] = None
    ):
        """
        Initialize LightGBM classifier.

        Args:
            params: LightGBM parameters (uses defaults if None)
            taxonomy_path: Path to taxonomy.json file
        """
        # Default LightGBM parameters
        self.default_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'verbose': -1
        }

        self.params = params if params is not None else self.default_params

        # Models for different hierarchy levels
        self.model_l1 = None
        self.model_l2 = None
        self.model_l3 = None

        # Label encoders
        self.label_encoder_l1 = LabelEncoder()
        self.label_encoder_l2 = LabelEncoder()
        self.label_encoder_l3 = LabelEncoder()

        # Taxonomy
        self.taxonomy = None
        if taxonomy_path:
            self.load_taxonomy(taxonomy_path)

        # Feature names
        self.feature_names = None

    def load_taxonomy(self, taxonomy_path: str):
        """
        Load category taxonomy from JSON file.

        Args:
            taxonomy_path: Path to taxonomy.json
        """
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)
        print(f"Loaded taxonomy with {len(self.taxonomy['categories'])} L1 categories")

    def prepare_features(
        self,
        embeddings: np.ndarray,
        enrichment_features: List[Dict]
    ) -> np.ndarray:
        """
        Combine embeddings with enrichment features.

        Args:
            embeddings: Sentence-BERT embeddings (n_samples, 384)
            enrichment_features: List of enrichment feature dictionaries

        Returns:
            Combined feature array
        """
        # For now, use embeddings as primary features
        # TODO: Add one-hot encoded enrichment features
        features = embeddings

        # Store feature names for explainability
        if self.feature_names is None:
            self.feature_names = [f'emb_{i}' for i in range(embeddings.shape[1])]

        return features

    def prepare_labels(
        self,
        transactions: List[Dict],
        level: str = 'l3'
    ) -> np.ndarray:
        """
        Extract and encode labels from transactions.

        Args:
            transactions: List of transactions with category labels
            level: Hierarchy level ('l1', 'l2', or 'l3')

        Returns:
            Encoded label array
        """
        # Extract labels
        labels = []
        for txn in transactions:
            if 'category' in txn:
                label = txn['category'].get(level, '')
            else:
                label = txn.get(level, '')
            labels.append(label)

        # Encode labels
        if level == 'l1':
            encoded = self.label_encoder_l1.fit_transform(labels)
        elif level == 'l2':
            encoded = self.label_encoder_l2.fit_transform(labels)
        else:  # l3
            encoded = self.label_encoder_l3.fit_transform(labels)

        return encoded

    def train(
        self,
        X: np.ndarray,
        y_l1: np.ndarray,
        y_l2: np.ndarray,
        y_l3: np.ndarray,
        validation_split: float = 0.15,
        num_boost_round: int = 100
    ) -> Dict[str, float]:
        """
        Train hierarchical models for L1, L2, and L3 classification.

        Args:
            X: Feature matrix
            y_l1: L1 labels (encoded)
            y_l2: L2 labels (encoded)
            y_l3: L3 labels (encoded)
            validation_split: Fraction of data for validation
            num_boost_round: Number of boosting rounds

        Returns:
            Dictionary with validation scores
        """
        print("Training hierarchical LightGBM models...")

        # Split data
        X_train, X_val, y_l1_train, y_l1_val = train_test_split(
            X, y_l1, test_size=validation_split, stratify=y_l1, random_state=42
        )
        _, _, y_l2_train, y_l2_val = train_test_split(
            X, y_l2, test_size=validation_split, stratify=y_l2, random_state=42
        )
        _, _, y_l3_train, y_l3_val = train_test_split(
            X, y_l3, test_size=validation_split, stratify=y_l3, random_state=42
        )

        scores = {}

        # Train L1 model
        print("\nTraining L1 classifier...")
        self.params['num_class'] = len(np.unique(y_l1))
        train_data_l1 = lgb.Dataset(X_train, label=y_l1_train)
        val_data_l1 = lgb.Dataset(X_val, label=y_l1_val, reference=train_data_l1)

        self.model_l1 = lgb.train(
            self.params,
            train_data_l1,
            num_boost_round=num_boost_round,
            valid_sets=[val_data_l1],
            valid_names=['validation']
        )
        scores['l1_accuracy'] = self._evaluate(X_val, y_l1_val, self.model_l1)

        # Train L2 model
        print("\nTraining L2 classifier...")
        self.params['num_class'] = len(np.unique(y_l2))
        train_data_l2 = lgb.Dataset(X_train, label=y_l2_train)
        val_data_l2 = lgb.Dataset(X_val, label=y_l2_val, reference=train_data_l2)

        self.model_l2 = lgb.train(
            self.params,
            train_data_l2,
            num_boost_round=num_boost_round,
            valid_sets=[val_data_l2],
            valid_names=['validation']
        )
        scores['l2_accuracy'] = self._evaluate(X_val, y_l2_val, self.model_l2)

        # Train L3 model
        print("\nTraining L3 classifier...")
        self.params['num_class'] = len(np.unique(y_l3))
        train_data_l3 = lgb.Dataset(X_train, label=y_l3_train)
        val_data_l3 = lgb.Dataset(X_val, label=y_l3_val, reference=train_data_l3)

        self.model_l3 = lgb.train(
            self.params,
            train_data_l3,
            num_boost_round=num_boost_round,
            valid_sets=[val_data_l3],
            valid_names=['validation']
        )
        scores['l3_accuracy'] = self._evaluate(X_val, y_l3_val, self.model_l3)

        print("\nTraining complete!")
        print(f"L1 Accuracy: {scores['l1_accuracy']:.4f}")
        print(f"L2 Accuracy: {scores['l2_accuracy']:.4f}")
        print(f"L3 Accuracy: {scores['l3_accuracy']:.4f}")

        return scores

    def _evaluate(self, X: np.ndarray, y_true: np.ndarray, model: lgb.Booster) -> float:
        """
        Evaluate model accuracy.

        Args:
            X: Feature matrix
            y_true: True labels
            model: Trained LightGBM model

        Returns:
            Accuracy score
        """
        y_pred_proba = model.predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = np.mean(y_pred == y_true)
        return accuracy

    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict categories for transactions.

        Args:
            X: Feature matrix
            return_proba: Return probability distributions

        Returns:
            Dictionary with predictions for each level
        """
        if self.model_l1 is None or self.model_l2 is None or self.model_l3 is None:
            raise ValueError("Models not trained. Call train() first.")

        predictions = {}

        # Predict L1
        l1_proba = self.model_l1.predict(X)
        l1_pred = np.argmax(l1_proba, axis=1)
        predictions['l1'] = self.label_encoder_l1.inverse_transform(l1_pred)
        if return_proba:
            predictions['l1_proba'] = l1_proba

        # Predict L2
        l2_proba = self.model_l2.predict(X)
        l2_pred = np.argmax(l2_proba, axis=1)
        predictions['l2'] = self.label_encoder_l2.inverse_transform(l2_pred)
        if return_proba:
            predictions['l2_proba'] = l2_proba

        # Predict L3
        l3_proba = self.model_l3.predict(X)
        l3_pred = np.argmax(l3_proba, axis=1)
        predictions['l3'] = self.label_encoder_l3.inverse_transform(l3_pred)
        if return_proba:
            predictions['l3_proba'] = l3_proba

        return predictions

    def get_feature_importance(self, level: str = 'l3', top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get feature importance for a specific hierarchy level.

        Args:
            level: Hierarchy level ('l1', 'l2', or 'l3')
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if level == 'l1':
            model = self.model_l1
        elif level == 'l2':
            model = self.model_l2
        else:
            model = self.model_l3

        if model is None:
            raise ValueError(f"Model for level {level} not trained")

        importance = model.feature_importance(importance_type='gain')
        feature_names = self.feature_names if self.feature_names else [f'f{i}' for i in range(len(importance))]

        # Sort by importance
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance[:top_k]

    def save_models(self, save_dir: str):
        """
        Save trained models and encoders.

        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM models
        self.model_l1.save_model(str(save_path / 'model_l1.txt'))
        self.model_l2.save_model(str(save_path / 'model_l2.txt'))
        self.model_l3.save_model(str(save_path / 'model_l3.txt'))

        # Save label encoders
        with open(save_path / 'encoders.pkl', 'wb') as f:
            pickle.dump({
                'l1': self.label_encoder_l1,
                'l2': self.label_encoder_l2,
                'l3': self.label_encoder_l3
            }, f)

        print(f"Models saved to: {save_path}")

    def load_models(self, model_dir: str):
        """
        Load trained models and encoders.

        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Load LightGBM models
        self.model_l1 = lgb.Booster(model_file=str(model_path / 'model_l1.txt'))
        self.model_l2 = lgb.Booster(model_file=str(model_path / 'model_l2.txt'))
        self.model_l3 = lgb.Booster(model_file=str(model_path / 'model_l3.txt'))

        # Load label encoders
        with open(model_path / 'encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
            self.label_encoder_l1 = encoders['l1']
            self.label_encoder_l2 = encoders['l2']
            self.label_encoder_l3 = encoders['l3']

        print(f"Models loaded from: {model_path}")


def main():
    """Example usage of LightGBMClassifier."""
    print("LightGBM Classifier initialized")
    print("Use this module with trained embeddings and labeled data")


if __name__ == "__main__":
    main()
