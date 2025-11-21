# Holmes AI - Financial Transaction Categorization Engine

An AI-native transaction categorization engine that converts unstructured bank transaction descriptions into structured, three-level hierarchical categories with high confidence scores.

## Features

- **Privacy-first**: On-premise processing with zero data exfiltration
- **Hierarchical Intelligence**: Three-level category taxonomy (L1 → L2 → L3)
- **Cost-effective**: Build and deploy for under $100 using open-source tools
- **Self-improving**: Continuous learning from user feedback

## Architecture

```
Raw Transaction → Data Ingestion → Pre-processing → Semantic Encoding → Classification → Confidence Scoring → Hierarchical Output
```

## Technology Stack

- **Semantic Encoding**: Sentence-BERT (all-MiniLM-L6-v2)
- **Classification**: LightGBM
- **Vector Database**: Supabase with pgvector
- **API**: FastAPI + Uvicorn
- **Monitoring**: Prometheus + Grafana

## Success Metrics

- Macro F1 score ≥ 0.90 at L3 categorization
- Inference latency < 200ms per transaction
- Misclassification rate < 2%
- Cost reduction of 65-75% vs. API-based solutions

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment configuration
cp .env.example .env

# 3. Train models (with your labeled data)
python train.py --data data/raw/transactions_train.csv

# 4. Run inference
python inference.py --data data/raw/transactions_test.csv --output results.json

# 5. Start API server
uvicorn src.api.main:app --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

## Documentation

- **[Setup Guide](SETUP.md)**: Detailed installation and configuration instructions
- **[PRD](HolmesAI%20PRD%20MD%20draft.md)**: Complete product requirements and technical specifications
- **[API Docs](http://localhost:8000/docs)**: Interactive API documentation (when server is running)

## Usage Examples

### Training Models

```bash
# Basic training
python train.py --data data/raw/labeled_transactions.csv

# Advanced training with custom parameters
python train.py \
  --data data/raw/labeled_transactions.csv \
  --output data/models \
  --rounds 150 \
  --validation-split 0.2
```

### Running Inference

```bash
# Command-line inference
python inference.py \
  --data data/raw/new_transactions.csv \
  --output results.json

# Results will include:
# - Hierarchical categories (L1, L2, L3)
# - Confidence scores
# - Review flags for low-confidence predictions
```

### Using the API

```python
import requests

# Categorize transactions via API
response = requests.post('http://localhost:8000/api/v1/categorize', json={
    "transactions": [{
        "transaction_id": "TXN_001",
        "merchant_raw": "SWIGGY*FOOD DELIVERY",
        "amount": 25.50,
        "currency": "USD",
        "timestamp": "2024-01-15T20:30:00Z",
        "mcc_code": "5814"
    }]
})

result = response.json()
print(result['results'][0]['category'])  # Predicted category
print(result['results'][0]['confidence'])  # Confidence score
```

### Docker Deployment

```bash
# Build and run with Docker
docker build -t holmes-ai .
docker run -p 8000:8000 holmes-ai

# Or use Docker Compose
docker-compose up -d
```

## Project Structure

```
Holmes_Cloe/
├── src/
│   ├── data_ingestion/     # Data loading and schema normalization
│   ├── preprocessing/      # Text cleaning and feature enrichment
│   ├── models/            # Sentence-BERT and LightGBM implementations
│   ├── api/               # FastAPI endpoints
│   ├── utils/             # Helper functions
│   └── config/            # Configuration files
├── data/
│   ├── raw/               # Raw transaction data
│   ├── processed/         # Processed features
│   └── models/            # Trained model artifacts
├── tests/                 # Unit and integration tests
└── requirements.txt       # Python dependencies
```

## Contact

- **Product Owner**: Pranav Mudigandur Venkat, Pratima Nemani
- **Engineering Lead**: Pranav Mudigandur Venkat, Pratima Nemani
