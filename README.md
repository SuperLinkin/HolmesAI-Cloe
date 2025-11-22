# Holmes AI - Financial Transaction Categorization Engine

An AI-native transaction categorization engine that converts unstructured bank transaction descriptions into structured, three-level hierarchical categories with high confidence scores.

## Features

- **Privacy-first**: On-premise processing with zero data exfiltration
- **Hierarchical Intelligence**: Three-level category taxonomy (L1 → L2 → L3)
- **Cost-effective**: Build and deploy for under $100 using open-source tools
- **Self-improving**: Continuous learning from user feedback

## Architecture

### 📊 Interactive Architecture Visualization

**🎨 NEW:** Explore the complete Holmes AI architecture with our interactive dashboard!

**[▶️ Open Architecture Dashboard](architecture_dashboard.html)** - Click to view animated workflow

Features:
- ✨ **Dual Workflows:** Switch between Inference Flow and Training Pipeline
- 🎬 **Animated Stages:** Watch data flow through the 6-stage pipeline
- 📊 **Live Metrics:** Real-time performance stats (10.2ms latency, 486 txns/sec)
- 🔍 **Stage Details:** Click any component for technical specifications
- 📱 **Fully Interactive:** Responsive design with smooth animations

**Quick Start:**
```bash
# Open in browser
start architecture_dashboard.html

# Or use a local server
python -m http.server 8000
# Then navigate to: http://localhost:8000/architecture_dashboard.html
```

See [ARCHITECTURE_DASHBOARD_README.md](ARCHITECTURE_DASHBOARD_README.md) for complete guide.

---

### Pipeline Overview

```
Raw Transaction → Data Ingestion → Pre-processing → Semantic Encoding → Classification → Confidence Scoring → Hierarchical Output
```

## Technology Stack

- **Semantic Encoding**: Sentence-BERT (all-mpnet-base-v2, 768D embeddings)
- **Classification**: LightGBM (500 boosting rounds, class-weighted)
- **Feature Engineering**: 5 engineered features + 768D embeddings = 773 total features
- **Vector Database**: Supabase with pgvector
- **API**: FastAPI + Uvicorn
- **Monitoring**: Prometheus + Grafana

## Achieved Results ✅

**Evaluation Date:** 2025-11-22
**Test Dataset:** 10,000 transactions
**Training Dataset:** 100,000 transactions

### Accuracy Metrics
- **L1 Macro F1:** 0.9960 (99.60%) - Target: ≥0.90 ✅
- **L2 Macro F1:** 0.9792 (97.92%) - Target: ≥0.90 ✅
- **L3 Macro F1:** 0.9728 (97.28%) - Target: ≥0.90 ✅

### Performance Metrics
- **Average Latency:** 10.22ms - Target: <200ms ✅ (19.5x faster!)
- **Throughput:** 486 txns/sec
- **Cost:** <$100 on-premise deployment ✅

**Status:** ALL TARGETS EXCEEDED - PRODUCTION READY ✅

For detailed results, see [FINAL_RESULTS_SUMMARY.md](FINAL_RESULTS_SUMMARY.md) and [evaluation_results/EVALUATION_REPORT.md](evaluation_results/EVALUATION_REPORT.md)

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

# 6. Open the Dashboard
cd frontend
open index.html  # or python -m http.server 8080
```

Visit the **Web Dashboard** at [frontend/index.html](frontend/index.html) or the **API Docs** at [http://localhost:8000/docs](http://localhost:8000/docs)

## 🎨 Web Dashboard - Now with Modern UI!

Holmes AI includes a **stunning, production-ready** web dashboard with:

### ✨ Latest Updates
- 🎨 **Modern Gradient Design**: Beautiful purple gradient background
- ⚡ **Real API Integration**: All mock data removed, live metrics only
- 💫 **Smooth Animations**: Fade-in, slide-in, hover effects throughout
- 🔄 **Interactive Elements**: Cards, buttons, and forms with delightful feedback
- 📱 **Fully Responsive**: Perfect on desktop, tablet, and mobile

### Features
- 📊 **Real-time Metrics**: Accuracy, latency, confidence from live API
- 🎯 **Live Categorization**: Test transactions with enhanced form inputs
- 🌳 **Taxonomy Browser**: Explore 15 L1 → 45+ L3 categories
- 📈 **Performance Charts**: Visual analytics with Chart.js
- 🎭 **Graceful Fallbacks**: Shows "N/A" when API is unavailable

**Quick Start**: Open `frontend/index.html` in your browser (API must be running)

## Documentation

- **[Web Dashboard](frontend/README.md)**: Frontend dashboard guide
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
