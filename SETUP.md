# Holmes AI Setup Guide

Complete setup guide for the Holmes AI transaction categorization engine.

## Prerequisites

- Python 3.10+
- pip
- Docker (optional, for containerized deployment)
- 8GB+ RAM recommended
- GPU optional (CPU works fine)

## Installation

### 1. Clone Repository

```bash
cd Holmes_Cloe
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example environment file
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edit .env with your configuration
# Minimum required: Set USE_MOCK_VECTOR_DB=True for development
```

## Data Preparation

### Expected Data Format

Your training data should be a CSV or JSON file with these fields:

```csv
transaction_id,merchant_raw,amount,currency,timestamp,channel,location,mcc_code,l1,l2,l3
TXN_001,AMZN MKTP US*2A3B4C5D6,49.99,USD,2024-01-15T14:32:00Z,online,Seattle WA,5942,Shopping,Shopping - Online,Shopping - Online - Amazon
```

**Required fields:**
- `transaction_id`: Unique identifier
- `merchant_raw`: Raw merchant description
- `amount`: Transaction amount
- `timestamp`: ISO 8601 timestamp

**Optional fields:**
- `currency`: ISO 4217 code (default: USD)
- `channel`: online, pos, atm
- `location`: Merchant location
- `mcc_code`: 4-digit MCC code

**Label fields (for training):**
- `l1`: Level 1 category
- `l2`: Level 2 category
- `l3`: Level 3 category

### Sample Data

Place your training data in `data/raw/`:

```bash
data/raw/
├── transactions_train.csv
└── transactions_test.csv
```

## Training Models

### Basic Training

```bash
python train.py --data data/raw/transactions_train.csv
```

### Advanced Training Options

```bash
python train.py \
  --data data/raw/transactions_train.csv \
  --output data/models \
  --rounds 100 \
  --validation-split 0.15
```

**Parameters:**
- `--data`: Path to labeled training data (required)
- `--output`: Output directory for models (default: data/models)
- `--rounds`: LightGBM boosting rounds (default: 100)
- `--validation-split`: Validation fraction (default: 0.15)

### Training Output

After training, you'll have:
```
data/models/
├── sentence_bert/      # Fine-tuned Sentence-BERT model
├── lightgbm/          # LightGBM classifiers (L1, L2, L3)
│   ├── model_l1.txt
│   ├── model_l2.txt
│   ├── model_l3.txt
│   └── encoders.pkl
└── training_metadata.json
```

## Running Inference

### Command Line

```bash
python inference.py \
  --data data/raw/transactions_test.csv \
  --output data/processed/results.json
```

### As API Server

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload

# API will be available at:
# - http://localhost:8000
# - Docs: http://localhost:8000/docs
```

### API Example

```bash
# Health check
curl http://localhost:8000/health

# Categorize transactions
curl -X POST http://localhost:8000/api/v1/categorize \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [{
      "transaction_id": "TXN_001",
      "merchant_raw": "SWIGGY*FOOD DELIVERY",
      "amount": 25.50,
      "currency": "USD",
      "timestamp": "2024-01-15T20:30:00Z",
      "channel": "online",
      "location": "Bangalore, KA",
      "mcc_code": "5814"
    }]
  }'
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t holmes-ai .

# Run container
docker run -p 8000:8000 holmes-ai
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f holmes-ai

# Stop services
docker-compose down
```

## Supabase Setup (Optional)

For production vector similarity search:

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Create new project
3. Get your project URL and anon key

### 2. Enable pgvector Extension

Run in Supabase SQL Editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Create Tables

```sql
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
```

### 4. Create Search Function

```sql
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
```

### 5. Update Environment

```bash
# In .env file:
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
USE_MOCK_VECTOR_DB=False
```

## Testing

### Module Tests

```python
# Test preprocessing
python -m src.preprocessing.text_cleaner

# Test feature enrichment
python -m src.preprocessing.feature_enrichment

# Test encoder
python -m src.models.sentence_bert_encoder
```

### API Tests

```bash
# Install test dependencies
pip install pytest requests

# Run tests (once implemented)
pytest tests/
```

## Performance Monitoring

### Metrics

- Inference latency: Target < 200ms per transaction
- Model accuracy: Target F1 ≥ 0.90
- Confidence distribution: High (≥85%), Medium (70-85%), Low (<70%)

### Logging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce batch size in training/inference
- Use CPU instead of GPU for smaller models

**2. Model Not Found**
- Ensure models are trained before inference
- Check paths in .env or command line arguments

**3. Import Errors**
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.10+)

**4. Slow Inference**
- Use GPU if available
- Increase batch size
- Consider model quantization

## Next Steps

1. **Prepare Training Data**: Label your transaction dataset
2. **Train Models**: Run `train.py` with your data
3. **Validate Performance**: Check F1 scores and accuracy
4. **Deploy API**: Start FastAPI server or use Docker
5. **Monitor**: Track confidence scores and review low-confidence predictions

## Support

- **Documentation**: See [README.md](README.md) and [PRD](HolmesAI%20PRD%20MD%20draft.md)
- **Issues**: Contact Pranav Mudigandur Venkat or Pratima Nemani
- **Email**: pranav@backbase.com, pratima@backbase.com
