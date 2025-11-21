"""
Sentence-BERT encoder for semantic representation of transactions.
"""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path


class SentenceBERTEncoder:
    """
    Encode transaction descriptions into dense semantic vectors using Sentence-BERT.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize Sentence-BERT encoder.

        Args:
            model_name: Name of the pre-trained Sentence-BERT model
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading Sentence-BERT model: {model_name}")
        print(f"Using device: {self.device}")

        # Load pre-trained model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) into semantic embeddings.

        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length (for cosine similarity)

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]

        # Encode using Sentence-BERT
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        return embeddings

    def encode_transactions(
        self,
        transactions: List[dict],
        text_field: str = 'merchant_cleaned',
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode a batch of transactions into embeddings.

        Args:
            transactions: List of transaction dictionaries
            text_field: Field to use for encoding (default: 'merchant_cleaned')
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        # Extract text from transactions
        texts = [txn.get(text_field, '') for txn in transactions]

        # Encode
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress=True
        )

        return embeddings

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)

    def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query embedding.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top similar candidates to return

        Returns:
            List of (index, similarity_score) tuples
        """
        # Compute similarities with all candidates
        similarities = np.dot(candidate_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return (index, score) pairs
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    def save_model(self, save_path: Union[str, Path]):
        """
        Save the fine-tuned model.

        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save(str(save_path))
        print(f"Model saved to: {save_path}")

    def load_model(self, model_path: Union[str, Path]):
        """
        Load a fine-tuned model.

        Args:
            model_path: Path to the saved model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        print(f"Loading model from: {model_path}")
        self.model = SentenceTransformer(str(model_path), device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_dim


def main():
    """Example usage of SentenceBERTEncoder."""

    # Initialize encoder
    encoder = SentenceBERTEncoder()

    # Test merchant descriptions
    test_merchants = [
        "amazon marketplace",
        "starbucks coffee",
        "uber trip",
        "swiggy food delivery",
        "netflix subscription"
    ]

    print("\nSemantic Encoding Example:")
    print("=" * 80)

    # Encode merchants
    print(f"\nEncoding {len(test_merchants)} merchants...")
    embeddings = encoder.encode(test_merchants, show_progress=False)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {encoder.get_embedding_dimension()}")

    # Compute similarities
    print("\nSimilarity Matrix:")
    print("-" * 80)

    for i, merchant1 in enumerate(test_merchants):
        print(f"\n{merchant1}:")
        for j, merchant2 in enumerate(test_merchants):
            if i != j:
                similarity = encoder.compute_similarity(embeddings[i], embeddings[j])
                print(f"  vs {merchant2:30s}: {similarity:.4f}")

    # Find similar merchants
    print("\n" + "=" * 80)
    query = "amazon online shopping"
    print(f"\nQuery: '{query}'")
    query_emb = encoder.encode(query)

    similar = encoder.find_similar(query_emb, embeddings, top_k=3)
    print(f"Top 3 similar merchants:")
    for idx, score in similar:
        print(f"  {test_merchants[idx]:30s} (similarity: {score:.4f})")


if __name__ == "__main__":
    main()
