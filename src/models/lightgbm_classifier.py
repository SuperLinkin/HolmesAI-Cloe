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
        # Enhanced LightGBM parameters for better accuracy
        self.default_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # Increased for more complex patterns
            'learning_rate': 0.03,  # Reduced for better generalization
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 10,  # Limit depth to prevent overfitting
            'min_data_in_leaf': 10,  # Reduced for better rare category handling
            'lambda_l1': 0.1,  # L1 regularization
            'lambda_l2': 0.1,  # L2 regularization
            'min_gain_to_split': 0.01,  # Minimum gain to make a split
            'verbose': -1,
            'force_col_wise': True  # Faster training on wide datasets
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

        # Hierarchy mappings for conditional prediction
        self.l1_to_l2_map = {}
        self.l2_to_l3_map = {}

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
        enrichment_features: List[Dict],
        include_enrichment: bool = True
    ) -> np.ndarray:
        """
        Combine embeddings with enrichment features.

        Args:
            embeddings: Sentence-BERT embeddings (n_samples, embedding_dim)
            enrichment_features: List of enrichment feature dictionaries
            include_enrichment: Whether to include enrichment features

        Returns:
            Combined feature array
        """
        if not include_enrichment:
            return embeddings

        # Extract enrichment features
        additional_features = []

        for txn in enrichment_features:
            txn_features = []

            # Spend band (one-hot encoded: 0-4)
            spend_band = txn.get('spend_band', 'medium')
            spend_band_map = {'micro': 0, 'low': 1, 'medium': 2, 'high': 3, 'premium': 4}
            txn_features.append(spend_band_map.get(spend_band, 2))

            # Temporal pattern (one-hot encoded: 0-3)
            temporal = txn.get('temporal_pattern', 'irregular')
            temporal_map = {'daily': 0, 'weekly': 1, 'monthly': 2, 'irregular': 3}
            txn_features.append(temporal_map.get(temporal, 3))

            # Channel (one-hot encoded: 0-3)
            channel = txn.get('channel', 'online')
            channel_map = {'online': 0, 'pos': 1, 'atm': 2, 'mobile': 3}
            txn_features.append(channel_map.get(channel, 0))

            # MCC code (normalized to 0-1 range)
            mcc = txn.get('mcc_code', 0)
            if mcc:
                mcc_normalized = min(int(mcc) / 10000.0, 1.0)  # Normalize MCC codes
            else:
                mcc_normalized = 0.0
            txn_features.append(mcc_normalized)

            # Amount percentile (normalized: 0-1)
            amount_percentile = txn.get('amount_percentile', 0.5)
            txn_features.append(float(amount_percentile) if amount_percentile else 0.5)

            additional_features.append(txn_features)

        # Combine embeddings with enrichment features
        additional_features_array = np.array(additional_features, dtype=np.float32)
        features = np.hstack([embeddings, additional_features_array])

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

    def build_hierarchy_maps(self, transactions: List[Dict]):
        """
        Build L1->L2 and L2->L3 mappings from training data.

        Args:
            transactions: List of transactions with category labels
        """
        for txn in transactions:
            if 'category' in txn:
                l1 = txn['category'].get('l1', '')
                l2 = txn['category'].get('l2', '')
                l3 = txn['category'].get('l3', '')
            else:
                l1 = txn.get('l1', '')
                l2 = txn.get('l2', '')
                l3 = txn.get('l3', '')

            if l1 and l2:
                if l1 not in self.l1_to_l2_map:
                    self.l1_to_l2_map[l1] = set()
                self.l1_to_l2_map[l1].add(l2)

            if l2 and l3:
                if l2 not in self.l2_to_l3_map:
                    self.l2_to_l3_map[l2] = set()
                self.l2_to_l3_map[l2].add(l3)

        # Convert sets to lists for easier use
        self.l1_to_l2_map = {k: list(v) for k, v in self.l1_to_l2_map.items()}
        self.l2_to_l3_map = {k: list(v) for k, v in self.l2_to_l3_map.items()}

        print(f"Built hierarchy maps: {len(self.l1_to_l2_map)} L1 categories, {len(self.l2_to_l3_map)} L2 categories")

    def train(
        self,
        X: np.ndarray,
        y_l1: np.ndarray,
        y_l2: np.ndarray,
        y_l3: np.ndarray,
        validation_split: float = 0.15,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        use_class_weight: bool = True
    ) -> Dict[str, float]:
        """
        Train hierarchical models for L1, L2, and L3 classification.

        Args:
            X: Feature matrix
            y_l1: L1 labels (encoded)
            y_l2: L2 labels (encoded)
            y_l3: L3 labels (encoded)
            validation_split: Fraction of data for validation
            num_boost_round: Number of boosting rounds (default: 500)
            early_stopping_rounds: Early stopping patience (default: 50)
            use_class_weight: Use class weighting to handle imbalance

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

        # Compute class weights if requested
        def compute_class_weights(y):
            """Compute balanced class weights."""
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            weight_dict = {i: w for i, w in enumerate(weights)}
            sample_weights = np.array([weight_dict[label] for label in y])
            return sample_weights

        # Train L1 model
        print("\nTraining L1 classifier...")
        self.params['num_class'] = len(np.unique(y_l1))

        if use_class_weight:
            l1_weights = compute_class_weights(y_l1_train)
            train_data_l1 = lgb.Dataset(X_train, label=y_l1_train, weight=l1_weights)
        else:
            train_data_l1 = lgb.Dataset(X_train, label=y_l1_train)

        val_data_l1 = lgb.Dataset(X_val, label=y_l1_val, reference=train_data_l1)

        self.model_l1 = lgb.train(
            self.params,
            train_data_l1,
            num_boost_round=num_boost_round,
            valid_sets=[val_data_l1],
            valid_names=['validation'],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        )
        scores['l1_accuracy'] = self._evaluate(X_val, y_l1_val, self.model_l1)

        # Train L2 model
        print("\nTraining L2 classifier...")
        self.params['num_class'] = len(np.unique(y_l2))

        if use_class_weight:
            l2_weights = compute_class_weights(y_l2_train)
            train_data_l2 = lgb.Dataset(X_train, label=y_l2_train, weight=l2_weights)
        else:
            train_data_l2 = lgb.Dataset(X_train, label=y_l2_train)

        val_data_l2 = lgb.Dataset(X_val, label=y_l2_val, reference=train_data_l2)

        self.model_l2 = lgb.train(
            self.params,
            train_data_l2,
            num_boost_round=num_boost_round,
            valid_sets=[val_data_l2],
            valid_names=['validation'],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        )
        scores['l2_accuracy'] = self._evaluate(X_val, y_l2_val, self.model_l2)

        # Train L3 model
        print("\nTraining L3 classifier...")
        self.params['num_class'] = len(np.unique(y_l3))

        if use_class_weight:
            l3_weights = compute_class_weights(y_l3_train)
            train_data_l3 = lgb.Dataset(X_train, label=y_l3_train, weight=l3_weights)
        else:
            train_data_l3 = lgb.Dataset(X_train, label=y_l3_train)

        val_data_l3 = lgb.Dataset(X_val, label=y_l3_val, reference=train_data_l3)

        self.model_l3 = lgb.train(
            self.params,
            train_data_l3,
            num_boost_round=num_boost_round,
            valid_sets=[val_data_l3],
            valid_names=['validation'],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
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
        return_proba: bool = True,
        use_hierarchy: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict categories for transactions with hierarchical constraints.

        Args:
            X: Feature matrix
            return_proba: Return probability distributions
            use_hierarchy: Apply hierarchical filtering (recommended)

        Returns:
            Dictionary with predictions for each level
        """
        if self.model_l1 is None or self.model_l2 is None or self.model_l3 is None:
            raise ValueError("Models not trained. Call train() first.")

        predictions = {}
        n_samples = X.shape[0]

        # Step 1: Predict L1
        l1_proba = self.model_l1.predict(X)
        l1_pred = np.argmax(l1_proba, axis=1)
        predictions['l1'] = self.label_encoder_l1.inverse_transform(l1_pred)
        if return_proba:
            predictions['l1_proba'] = l1_proba

        # Step 2: Predict L2 with hierarchical filtering
        l2_proba = self.model_l2.predict(X)
        l2_pred = np.zeros(n_samples, dtype=int)

        if use_hierarchy and self.l1_to_l2_map:
            # For each sample, filter L2 predictions based on L1
            for i in range(n_samples):
                l1_category = predictions['l1'][i]
                valid_l2_categories = self.l1_to_l2_map.get(l1_category, [])

                if valid_l2_categories:
                    # Get indices of valid L2 categories
                    valid_l2_indices = []
                    for l2_cat in valid_l2_categories:
                        try:
                            idx = np.where(self.label_encoder_l2.classes_ == l2_cat)[0]
                            if len(idx) > 0:
                                valid_l2_indices.append(idx[0])
                        except:
                            continue

                    if valid_l2_indices:
                        # Mask out invalid L2 categories by setting their probabilities to -inf
                        masked_proba = np.full(l2_proba.shape[1], -np.inf)
                        masked_proba[valid_l2_indices] = l2_proba[i, valid_l2_indices]
                        l2_pred[i] = np.argmax(masked_proba)
                    else:
                        l2_pred[i] = np.argmax(l2_proba[i])
                else:
                    l2_pred[i] = np.argmax(l2_proba[i])
        else:
            l2_pred = np.argmax(l2_proba, axis=1)

        predictions['l2'] = self.label_encoder_l2.inverse_transform(l2_pred)
        if return_proba:
            predictions['l2_proba'] = l2_proba

        # Step 3: Predict L3 with hierarchical filtering
        l3_proba = self.model_l3.predict(X)
        l3_pred = np.zeros(n_samples, dtype=int)

        if use_hierarchy and self.l2_to_l3_map:
            # For each sample, filter L3 predictions based on L2
            for i in range(n_samples):
                l2_category = predictions['l2'][i]
                valid_l3_categories = self.l2_to_l3_map.get(l2_category, [])

                if valid_l3_categories:
                    # Get indices of valid L3 categories
                    valid_l3_indices = []
                    for l3_cat in valid_l3_categories:
                        try:
                            idx = np.where(self.label_encoder_l3.classes_ == l3_cat)[0]
                            if len(idx) > 0:
                                valid_l3_indices.append(idx[0])
                        except:
                            continue

                    if valid_l3_indices:
                        # Mask out invalid L3 categories
                        masked_proba = np.full(l3_proba.shape[1], -np.inf)
                        masked_proba[valid_l3_indices] = l3_proba[i, valid_l3_indices]
                        l3_pred[i] = np.argmax(masked_proba)
                    else:
                        l3_pred[i] = np.argmax(l3_proba[i])
                else:
                    l3_pred[i] = np.argmax(l3_proba[i])
        else:
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

        # Save label encoders and hierarchy maps
        with open(save_path / 'encoders.pkl', 'wb') as f:
            pickle.dump({
                'l1': self.label_encoder_l1,
                'l2': self.label_encoder_l2,
                'l3': self.label_encoder_l3,
                'l1_to_l2_map': self.l1_to_l2_map,
                'l2_to_l3_map': self.l2_to_l3_map
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

        # Load label encoders and hierarchy maps
        with open(model_path / 'encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
            self.label_encoder_l1 = encoders['l1']
            self.label_encoder_l2 = encoders['l2']
            self.label_encoder_l3 = encoders['l3']
            # Load hierarchy maps if available (for backward compatibility)
            self.l1_to_l2_map = encoders.get('l1_to_l2_map', {})
            self.l2_to_l3_map = encoders.get('l2_to_l3_map', {})

        print(f"Models loaded from: {model_path}")


def main():
    """Example usage of LightGBMClassifier."""
    print("LightGBM Classifier initialized")
    print("Use this module with trained embeddings and labeled data")


if __name__ == "__main__":
    main()
