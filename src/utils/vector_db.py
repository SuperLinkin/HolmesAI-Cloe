"""
Vector database integration with Supabase and pgvector.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from supabase import create_client, Client
import os


class VectorDatabase:
    """
    Interface for vector database operations using Supabase with pgvector.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None
    ):
        """
        Initialize vector database connection.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
        """
        # Get credentials from environment if not provided
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')

        if not self.supabase_url or not self.supabase_key:
            print("Warning: Supabase credentials not provided. Vector DB features disabled.")
            self.client = None
        else:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("Connected to Supabase vector database")

        self.table_name = "merchant_embeddings"

    def create_table(self):
        """
        Create the merchant embeddings table with pgvector extension.

        SQL Schema:
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE merchant_embeddings (
            id SERIAL PRIMARY KEY,
            merchant_name TEXT NOT NULL,
            merchant_cleaned TEXT NOT NULL,
            embedding VECTOR(384),
            category_l1 TEXT,
            category_l2 TEXT,
            category_l3 TEXT,
            category_l3_id TEXT,
            transaction_count INTEGER DEFAULT 1,
            last_seen TIMESTAMP DEFAULT NOW(),
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX ON merchant_embeddings USING hnsw (embedding vector_cosine_ops);
        """
        print("Table creation should be done via Supabase SQL editor.")
        print("See docstring for SQL schema.")

    def insert_merchant(
        self,
        merchant_name: str,
        merchant_cleaned: str,
        embedding: np.ndarray,
        category_l1: str,
        category_l2: str,
        category_l3: str,
        category_l3_id: str
    ) -> Dict:
        """
        Insert a new merchant embedding into the database.

        Args:
            merchant_name: Raw merchant name
            merchant_cleaned: Cleaned merchant name
            embedding: Semantic embedding vector
            category_l1: L1 category
            category_l2: L2 category
            category_l3: L3 category
            category_l3_id: L3 category ID

        Returns:
            Inserted record
        """
        if self.client is None:
            raise ValueError("Supabase client not initialized")

        data = {
            "merchant_name": merchant_name,
            "merchant_cleaned": merchant_cleaned,
            "embedding": embedding.tolist(),
            "category_l1": category_l1,
            "category_l2": category_l2,
            "category_l3": category_l3,
            "category_l3_id": category_l3_id
        }

        response = self.client.table(self.table_name).insert(data).execute()
        return response.data[0] if response.data else {}

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar merchants using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of similar merchant records with similarity scores
        """
        if self.client is None:
            raise ValueError("Supabase client not initialized")

        # Note: This requires a custom RPC function in Supabase
        # See create_similarity_search_function() for setup

        response = self.client.rpc(
            'search_similar_merchants',
            {
                'query_embedding': query_embedding.tolist(),
                'match_count': top_k,
                'similarity_threshold': similarity_threshold
            }
        ).execute()

        return response.data if response.data else []

    def get_merchant_by_name(self, merchant_cleaned: str) -> Optional[Dict]:
        """
        Get merchant record by cleaned name.

        Args:
            merchant_cleaned: Cleaned merchant name

        Returns:
            Merchant record if found, None otherwise
        """
        if self.client is None:
            return None

        response = self.client.table(self.table_name)\
            .select("*")\
            .eq("merchant_cleaned", merchant_cleaned)\
            .execute()

        return response.data[0] if response.data else None

    def update_transaction_count(self, merchant_id: int):
        """
        Increment transaction count for a merchant.

        Args:
            merchant_id: Merchant record ID
        """
        if self.client is None:
            return

        self.client.table(self.table_name)\
            .update({"transaction_count": "transaction_count + 1"})\
            .eq("id", merchant_id)\
            .execute()

    def get_merchant_aliases(self, category_l3_id: str) -> List[str]:
        """
        Get all known merchant aliases for a category.

        Args:
            category_l3_id: L3 category ID

        Returns:
            List of merchant aliases
        """
        if self.client is None:
            return []

        response = self.client.table(self.table_name)\
            .select("merchant_cleaned")\
            .eq("category_l3_id", category_l3_id)\
            .execute()

        return [record["merchant_cleaned"] for record in response.data]

    def bulk_insert_merchants(self, merchants: List[Dict]) -> int:
        """
        Bulk insert multiple merchant records.

        Args:
            merchants: List of merchant dictionaries with required fields

        Returns:
            Number of records inserted
        """
        if self.client is None:
            raise ValueError("Supabase client not initialized")

        # Convert numpy arrays to lists
        for merchant in merchants:
            if isinstance(merchant.get('embedding'), np.ndarray):
                merchant['embedding'] = merchant['embedding'].tolist()

        response = self.client.table(self.table_name).insert(merchants).execute()
        return len(response.data) if response.data else 0

    def create_similarity_search_function(self):
        """
        Create PostgreSQL function for similarity search.

        This SQL should be run in Supabase SQL editor:

        CREATE OR REPLACE FUNCTION search_similar_merchants(
            query_embedding VECTOR(384),
            match_count INT DEFAULT 10,
            similarity_threshold FLOAT DEFAULT 0.7
        )
        RETURNS TABLE (
            id INT,
            merchant_name TEXT,
            merchant_cleaned TEXT,
            category_l1 TEXT,
            category_l2 TEXT,
            category_l3 TEXT,
            category_l3_id TEXT,
            similarity FLOAT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                me.id,
                me.merchant_name,
                me.merchant_cleaned,
                me.category_l1,
                me.category_l2,
                me.category_l3,
                me.category_l3_id,
                1 - (me.embedding <=> query_embedding) AS similarity
            FROM merchant_embeddings me
            WHERE 1 - (me.embedding <=> query_embedding) >= similarity_threshold
            ORDER BY me.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        """
        print("Similarity search function should be created via Supabase SQL editor.")
        print("See docstring for SQL function definition.")


class MockVectorDatabase:
    """
    Mock vector database for development without Supabase.
    """

    def __init__(self):
        self.merchants = []
        self.embeddings = []
        print("Using mock vector database (no Supabase connection)")

    def insert_merchant(self, merchant_name: str, merchant_cleaned: str,
                       embedding: np.ndarray, category_l1: str,
                       category_l2: str, category_l3: str,
                       category_l3_id: str) -> Dict:
        """Insert merchant into mock database."""
        record = {
            "id": len(self.merchants),
            "merchant_name": merchant_name,
            "merchant_cleaned": merchant_cleaned,
            "embedding": embedding,
            "category_l1": category_l1,
            "category_l2": category_l2,
            "category_l3": category_l3,
            "category_l3_id": category_l3_id
        }
        self.merchants.append(record)
        self.embeddings.append(embedding)
        return record

    def search_similar(self, query_embedding: np.ndarray,
                      top_k: int = 10,
                      similarity_threshold: float = 0.7) -> List[Dict]:
        """Search for similar merchants in mock database."""
        if not self.embeddings:
            return []

        # Compute cosine similarities
        embeddings_array = np.array(self.embeddings)
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )

        # Filter by threshold and get top-k
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        if len(valid_indices) == 0:
            return []

        top_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k:][::-1]]

        results = []
        for idx in top_indices:
            record = self.merchants[idx].copy()
            record['similarity'] = float(similarities[idx])
            results.append(record)

        return results

    def get_merchant_by_name(self, merchant_cleaned: str) -> Optional[Dict]:
        """Get merchant by name from mock database."""
        for merchant in self.merchants:
            if merchant["merchant_cleaned"] == merchant_cleaned:
                return merchant
        return None


def get_vector_db(use_mock: bool = False) -> VectorDatabase:
    """
    Factory function to get appropriate vector database instance.

    Args:
        use_mock: If True, return mock database

    Returns:
        Vector database instance
    """
    if use_mock or not os.getenv('SUPABASE_URL'):
        return MockVectorDatabase()
    else:
        return VectorDatabase()


def main():
    """Example usage of VectorDatabase."""
    print("Vector Database Module")
    print("=" * 80)
    print("\nTo use Supabase vector database:")
    print("1. Set SUPABASE_URL and SUPABASE_KEY environment variables")
    print("2. Run SQL schema from create_table() docstring")
    print("3. Run SQL function from create_similarity_search_function() docstring")
    print("\nOr use MockVectorDatabase for development without Supabase")


if __name__ == "__main__":
    main()
