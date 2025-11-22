"""
Script to populate Supabase database with transaction embeddings.

This script:
1. Loads the trained Sentence-BERT model
2. Processes transactions from the dataset
3. Generates embeddings for each transaction
4. Stores embeddings in Supabase with metadata
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

from src.models.sentence_bert_encoder import SentenceBERTEncoder
from src.preprocessing.preprocessor import TransactionPreprocessor
from src.data_ingestion import DataIngestion

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def main():
    """Main function to populate database."""

    print("=" * 80)
    print("HOLMES AI - DATABASE POPULATION")
    print("=" * 80)
    print()

    # Check environment variables
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[ERROR] Missing Supabase credentials in .env file")
        print("Please ensure SUPABASE_URL and SUPABASE_KEY are set")
        return

    print(f"[OK] Supabase URL: {SUPABASE_URL[:30]}...")
    print()

    # Initialize Supabase client
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[OK] Connected to Supabase")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Supabase: {e}")
        return

    # Load trained Sentence-BERT model
    encoder_path = Path("data/models/sentence_bert")
    if not encoder_path.exists():
        print(f"[ERROR] Trained model not found at {encoder_path}")
        print("Please run training first: python train.py")
        return

    print(f"[OK] Loading Sentence-BERT model from {encoder_path}")
    try:
        # The saved model is a SentenceTransformer directory
        encoder = SentenceBERTEncoder(model_name=str(encoder_path))
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Load and preprocess transactions
    data_path = "data/synthetic_transactions_1k.csv"
    print(f"\n[OK] Loading transactions from {data_path}")

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(df)} transactions")

    # Process transactions
    ingestion = DataIngestion()
    normalized = ingestion.ingest_pipeline(data_path)

    preprocessor = TransactionPreprocessor()
    preprocessed = preprocessor.preprocess_batch([txn.model_dump() for txn in normalized])

    print(f"[OK] Preprocessed {len(preprocessed)} transactions")

    # Generate embeddings
    print("\n[INFO] Generating embeddings...")
    embeddings = encoder.encode_transactions(
        preprocessed,
        text_field='merchant_cleaned',
        batch_size=32
    )
    print(f"[OK] Generated {embeddings.shape[0]} embeddings (dimension: {embeddings.shape[1]})")

    # Prepare data for insertion (aggregate by merchant)
    print("\n[INFO] Aggregating transactions by merchant...")

    # Group by merchant_cleaned to get unique merchants
    from collections import defaultdict
    merchant_data = defaultdict(lambda: {
        'embeddings': [],
        'categories': {'l1': [], 'l2': [], 'l3': [], 'l3_id': []},
        'count': 0,
        'merchant_raw': None,
        'last_seen': None
    })

    for i, (txn, embedding) in enumerate(zip(preprocessed, embeddings)):
        orig_row = df.iloc[i]
        merchant_key = txn['merchant_cleaned']

        merchant_data[merchant_key]['embeddings'].append(embedding)
        merchant_data[merchant_key]['categories']['l1'].append(orig_row['l1'])
        merchant_data[merchant_key]['categories']['l2'].append(orig_row['l2'])
        merchant_data[merchant_key]['categories']['l3'].append(orig_row['l3'])
        merchant_data[merchant_key]['categories']['l3_id'].append(orig_row['l3_id'])
        merchant_data[merchant_key]['count'] += 1
        merchant_data[merchant_key]['merchant_raw'] = txn['merchant_raw']

        # Convert timestamp to string (handle pandas Timestamp)
        ts = txn.get('timestamp', datetime.now())
        if isinstance(ts, str):
            merchant_data[merchant_key]['last_seen'] = ts
        else:
            merchant_data[merchant_key]['last_seen'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

    print(f"[OK] Found {len(merchant_data)} unique merchants")

    # Create records for insertion
    records = []
    for merchant_cleaned, data in merchant_data.items():
        # Average the embeddings for this merchant
        avg_embedding = np.mean(data['embeddings'], axis=0)

        # Get most common category (mode)
        from collections import Counter
        most_common_l1 = Counter(data['categories']['l1']).most_common(1)[0][0]
        most_common_l2 = Counter(data['categories']['l2']).most_common(1)[0][0]
        most_common_l3 = Counter(data['categories']['l3']).most_common(1)[0][0]
        most_common_l3_id = Counter(data['categories']['l3_id']).most_common(1)[0][0]

        record = {
            "merchant_name": data['merchant_raw'],
            "merchant_cleaned": merchant_cleaned,
            "embedding": avg_embedding.tolist(),
            "category_l1": most_common_l1,
            "category_l2": most_common_l2,
            "category_l3": most_common_l3,
            "category_l3_id": most_common_l3_id,
            "transaction_count": data['count'],
            "last_seen": data['last_seen'],
            "created_at": datetime.now().isoformat()
        }
        records.append(record)

    print(f"[OK] Prepared {len(records)} unique merchant records for insertion")

    # Insert in batches
    batch_size = 100
    total_inserted = 0

    print(f"\n[INFO] Inserting records in batches of {batch_size}...")

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        try:
            result = supabase.table("merchant_embeddings").insert(batch).execute()
            total_inserted += len(batch)
            print(f"[OK] Inserted batch {i//batch_size + 1}: {len(batch)} records (Total: {total_inserted}/{len(records)})")
        except Exception as e:
            print(f"[ERROR] Failed to insert batch {i//batch_size + 1}: {e}")
            print("Continuing with next batch...")

    print()
    print("=" * 80)
    print("DATABASE POPULATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal records inserted: {total_inserted}/{len(records)}")
    print(f"Success rate: {total_inserted/len(records)*100:.1f}%")
    print()
    print("You can now:")
    print("1. View the data in Supabase dashboard")
    print("2. Use vector similarity search for similar transactions")
    print("3. Query by category, merchant, or amount")
    print()

if __name__ == "__main__":
    main()
