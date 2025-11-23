# Holmes AI - Deployment Guide

## Overview

This guide provides multiple deployment options for Holmes AI, from quick on-premise installations to scalable cloud deployments.

---

## 🚀 Deployment Options

### **Option 1: Docker Deployment (Recommended)**

**Best for:** Quick deployment, consistent environments, easy scaling

#### Prerequisites
- Docker installed on the target system
- 4GB RAM minimum, 8GB recommended
- 10GB disk space

#### Step 1: Create Dockerfile

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY train.py .
COPY demo.py .

# Expose port for API (if using FastAPI)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import src.models.lightgbm_classifier as lgb; print('OK')"

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 2: Create docker-compose.yml

```yaml
version: '3.8'

services:
  holmes-ai:
    build: .
    container_name: holmes-ai
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/models
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    restart: unless-stopped
    mem_limit: 4g
    cpus: 2
```

#### Step 3: Deploy

```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f holmes-ai

# Stop
docker-compose down
```

#### Step 4: Test Deployment

```bash
curl -X POST http://localhost:8000/categorize \
  -H "Content-Type: application/json" \
  -d '{
    "merchant": "STARBUCKS #4532",
    "amount": 5.25,
    "date": "2025-01-15",
    "mcc_code": 5812
  }'
```

---

### **Option 2: On-Premise Python Installation**

**Best for:** Clients with existing Python infrastructure, full control

#### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum
- Linux/Windows/macOS

#### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/holmes-ai.git
cd holmes-ai
```

#### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

#### Step 3: Download Pre-trained Models

```bash
# Option A: Download from release
wget https://github.com/your-org/holmes-ai/releases/download/v1.0/models.zip
unzip models.zip

# Option B: Train from scratch
python train.py \
  --data data/synthetic_transactions_100k.csv \
  --output models \
  --rounds 500 \
  --validation-split 0.15
```

#### Step 4: Run as Service

**Linux (systemd):**

Create `/etc/systemd/system/holmes-ai.service`:

```ini
[Unit]
Description=Holmes AI Transaction Categorization Service
After=network.target

[Service]
Type=simple
User=holmesai
WorkingDirectory=/opt/holmes-ai
ExecStart=/opt/holmes-ai/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable holmes-ai
sudo systemctl start holmes-ai
sudo systemctl status holmes-ai
```

**Windows (NSSM):**

```powershell
# Download NSSM
# Install as service
nssm install HolmesAI "C:\holmes-ai\venv\Scripts\python.exe" "C:\holmes-ai\venv\Scripts\uvicorn" "src.api.main:app" "--host" "0.0.0.0" "--port" "8000"
nssm start HolmesAI
```

---

### **Option 3: Cloud Deployment (AWS/GCP/Azure)**

**Best for:** Scalability, high availability, managed infrastructure

#### AWS Deployment (ECS + Fargate)

**Step 1: Build and Push Docker Image**

```bash
# Build image
docker build -t holmes-ai:latest .

# Tag for ECR
docker tag holmes-ai:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/holmes-ai:latest

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/holmes-ai:latest
```

**Step 2: Create ECS Task Definition**

```json
{
  "family": "holmes-ai-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "holmes-ai",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/holmes-ai:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/models"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/holmes-ai",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Step 3: Create ECS Service**

```bash
aws ecs create-service \
  --cluster holmes-ai-cluster \
  --service-name holmes-ai-service \
  --task-definition holmes-ai-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=holmes-ai,containerPort=8000"
```

#### Google Cloud Platform (Cloud Run)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/holmes-ai

# Deploy to Cloud Run
gcloud run deploy holmes-ai \
  --image gcr.io/PROJECT-ID/holmes-ai \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated
```

#### Azure (Container Instances)

```bash
# Build and push to ACR
az acr build --registry holmesai --image holmes-ai:latest .

# Deploy to ACI
az container create \
  --resource-group holmes-ai-rg \
  --name holmes-ai \
  --image holmesai.azurecr.io/holmes-ai:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --dns-name-label holmes-ai \
  --environment-variables MODEL_PATH=/app/models
```

---

### **Option 4: Kubernetes Deployment**

**Best for:** Enterprise clients, multi-tenancy, auto-scaling

#### Kubernetes Manifests

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: holmes-ai
  labels:
    app: holmes-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: holmes-ai
  template:
    metadata:
      labels:
        app: holmes-ai
    spec:
      containers:
      - name: holmes-ai
        image: your-registry/holmes-ai:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: MODEL_PATH
          value: "/app/models"
        - name: LOG_LEVEL
          value: "INFO"
```

**service.yaml:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: holmes-ai-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: holmes-ai
```

**hpa.yaml (Auto-scaling):**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: holmes-ai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: holmes-ai
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Deploy:**

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml

# Check status
kubectl get pods
kubectl get svc holmes-ai-service
```

---

## 🔧 Configuration Management

### Environment Variables

```bash
# Model configuration
MODEL_PATH=/app/models
TAXONOMY_PATH=/app/src/config/taxonomy.json

# Performance tuning
MAX_WORKERS=4
BATCH_SIZE=32
CACHE_SIZE=1000

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/holmes-ai.log

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_TIMEOUT=30
```

### Configuration File (config.yaml)

```yaml
model:
  path: "/app/models"
  l1_model: "lightgbm_l1.txt"
  l2_model: "lightgbm_l2.txt"
  l3_model: "lightgbm_l3.txt"
  encoder: "sentence_bert_encoder.pkl"

taxonomy:
  path: "/app/src/config/taxonomy.json"
  reload_interval: 300  # seconds

performance:
  max_workers: 4
  batch_size: 32
  embedding_cache_size: 1000
  use_gpu: false

logging:
  level: "INFO"
  format: "json"
  file: "/app/logs/holmes-ai.log"
  max_size_mb: 100
  backup_count: 5

api:
  host: "0.0.0.0"
  port: 8000
  timeout: 30
  cors_origins: ["*"]
```

---

## 📊 Monitoring & Observability

### Health Check Endpoint

```python
# src/api/main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": True
    }

@app.get("/ready")
async def readiness_check():
    # Check if models are loaded
    # Check database connectivity
    return {"status": "ready"}
```

### Metrics (Prometheus)

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
categorization_total = Counter('categorization_total', 'Total categorizations')
categorization_errors = Counter('categorization_errors', 'Total errors')

# Histograms
categorization_latency = Histogram('categorization_latency_seconds', 'Categorization latency')
confidence_score = Histogram('confidence_score', 'Confidence scores')

# Gauges
model_memory_usage = Gauge('model_memory_mb', 'Model memory usage in MB')
```

### Logging (Structured JSON)

```python
import logging
import json

logger = logging.getLogger(__name__)

def log_categorization(transaction, result, latency):
    logger.info(json.dumps({
        "event": "categorization",
        "merchant": transaction["merchant"],
        "amount": transaction["amount"],
        "predicted_l1": result["L1"],
        "predicted_l2": result["L2"],
        "predicted_l3": result["L3"],
        "confidence": result["confidence"],
        "latency_ms": latency * 1000,
        "timestamp": datetime.utcnow().isoformat()
    }))
```

---

## 🔐 Security Considerations

### API Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.post("/categorize")
async def categorize(transaction: dict, token: str = Security(verify_token)):
    # Process transaction
    pass
```

### HTTPS/TLS

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with HTTPS
uvicorn src.api.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/categorize")
@limiter.limit("100/minute")
async def categorize(request: Request, transaction: dict):
    # Process transaction
    pass
```

---

## 💾 Database Integration (Optional)

### PostgreSQL for Transaction Logging

```python
import psycopg2
from psycopg2.extras import execute_values

def log_to_database(transactions):
    conn = psycopg2.connect(
        host="localhost",
        database="holmes_ai",
        user="postgres",
        password="password"
    )

    cursor = conn.cursor()

    query = """
        INSERT INTO categorizations
        (merchant, amount, date, predicted_l1, predicted_l2, predicted_l3, confidence, created_at)
        VALUES %s
    """

    values = [(t["merchant"], t["amount"], t["date"],
               t["result"]["L1"], t["result"]["L2"], t["result"]["L3"],
               t["result"]["confidence"], datetime.utcnow())
              for t in transactions]

    execute_values(cursor, query, values)
    conn.commit()
    cursor.close()
    conn.close()
```

---

## 📦 Client Integration Examples

### Python Client

```python
import requests

class HolmesAIClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def categorize(self, merchant, amount, date, mcc_code=None):
        response = requests.post(
            f"{self.base_url}/categorize",
            json={
                "merchant": merchant,
                "amount": amount,
                "date": date,
                "mcc_code": mcc_code
            },
            headers=self.headers
        )
        return response.json()

# Usage
client = HolmesAIClient(base_url="https://holmes-ai.example.com")
result = client.categorize("STARBUCKS #4532", 5.25, "2025-01-15", 5812)
print(f"Category: {result['L1']} → {result['L2']} → {result['L3']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class HolmesAIClient {
  constructor(baseURL = 'http://localhost:8000', apiKey = null) {
    this.client = axios.create({
      baseURL,
      headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}
    });
  }

  async categorize(merchant, amount, date, mccCode = null) {
    const response = await this.client.post('/categorize', {
      merchant,
      amount,
      date,
      mcc_code: mccCode
    });
    return response.data;
  }
}

// Usage
const client = new HolmesAIClient('https://holmes-ai.example.com');
const result = await client.categorize('STARBUCKS #4532', 5.25, '2025-01-15', 5812);
console.log(`Category: ${result.L1} → ${result.L2} → ${result.L3}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

### cURL Example

```bash
curl -X POST https://holmes-ai.example.com/categorize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "merchant": "STARBUCKS #4532",
    "amount": 5.25,
    "date": "2025-01-15",
    "mcc_code": 5812
  }'
```

---

## 🔄 Update & Maintenance

### Model Updates

```bash
# Download new model
wget https://releases.example.com/holmes-ai/models-v1.1.zip

# Backup old models
mv models models.backup

# Extract new models
unzip models-v1.1.zip

# Restart service (Docker)
docker-compose restart holmes-ai

# Restart service (systemd)
sudo systemctl restart holmes-ai
```

### Taxonomy Updates

```bash
# Edit taxonomy
vim src/config/taxonomy.json

# Validate taxonomy
python validate_taxonomy.py

# Reload without downtime (if hot-reload enabled)
curl -X POST http://localhost:8000/admin/reload-taxonomy \
  -H "Authorization: Bearer ADMIN_TOKEN"

# Or restart service
docker-compose restart holmes-ai
```

---

## 📈 Scaling Recommendations

| Monthly Transactions | Deployment | Resources | Cost Estimate |
|---------------------|------------|-----------|---------------|
| < 100K | Single Docker container | 2 CPU, 4GB RAM | $20-50/month |
| 100K - 1M | Load balanced (2-3 instances) | 2 CPU, 4GB RAM each | $100-200/month |
| 1M - 10M | Kubernetes cluster (5-10 pods) | 2 CPU, 4GB RAM per pod | $500-1000/month |
| > 10M | Auto-scaling K8s (10-50 pods) | 2 CPU, 4GB RAM per pod | $2000-5000/month |

---

## 🆘 Troubleshooting

### Common Issues

**Issue: Model not loading**
```bash
# Check model files exist
ls -lh models/

# Verify permissions
chmod -R 755 models/

# Check logs
docker-compose logs holmes-ai
```

**Issue: High latency**
```bash
# Check CPU/memory usage
docker stats holmes-ai

# Increase workers
export MAX_WORKERS=8

# Enable caching
export CACHE_SIZE=5000
```

**Issue: Out of memory**
```bash
# Increase Docker memory limit
docker update --memory 8g holmes-ai

# Or in docker-compose.yml
mem_limit: 8g
```

---

## 📞 Support

For deployment assistance:
- **Documentation**: [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md)
- **Issues**: GitHub Issues
- **Email**: support@holmesai.example.com

---

**Generated:** 2025-11-23
**Version:** 1.0
**Status:** Production Ready
