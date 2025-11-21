"""
FastAPI application for Holmes AI transaction categorization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import time
import numpy as np

from ..data_ingestion.schema import TransactionInput, TransactionOutput, CategoryPrediction
from ..preprocessing.preprocessor import TransactionPreprocessor
from ..models.sentence_bert_encoder import SentenceBERTEncoder
from ..models.lightgbm_classifier import LightGBMClassifier
from ..utils.confidence_scorer import ConfidenceScorer

# Initialize FastAPI app
app = FastAPI(
    title="Holmes AI - Transaction Categorization API",
    description="AI-native transaction categorization engine with hierarchical taxonomy",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class CategorizeRequest(BaseModel):
    """Request model for categorization endpoint."""
    transactions: List[TransactionInput]

    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_id": "TXN_001",
                        "merchant_raw": "AMZN MKTP US*2A3B4C5D6",
                        "amount": 49.99,
                        "currency": "USD",
                        "timestamp": "2024-01-15T14:32:00Z",
                        "channel": "online",
                        "location": "Seattle, WA",
                        "mcc_code": "5942"
                    }
                ]
            }
        }


class CategorizeResponse(BaseModel):
    """Response model for categorization endpoint."""
    results: List[TransactionOutput]
    metadata: Dict

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "transaction_id": "TXN_001",
                        "category": {
                            "l1": "Shopping",
                            "l1_id": "SHP",
                            "l2": "Shopping - Online",
                            "l2_id": "SHP-ONL",
                            "l3": "Shopping - Online - Amazon",
                            "l3_id": "SHP-ONL-AMZ"
                        },
                        "confidence": 0.94,
                        "processing_time_ms": 145.0,
                        "explanation": {}
                    }
                ],
                "metadata": {
                    "total_processed": 1,
                    "avg_confidence": 0.94,
                    "low_confidence_count": 0
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_loaded: bool


# Global model instances (loaded on startup)
preprocessor: Optional[TransactionPreprocessor] = None
encoder: Optional[SentenceBERTEncoder] = None
classifier: Optional[LightGBMClassifier] = None
confidence_scorer: Optional[ConfidenceScorer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on application startup."""
    global preprocessor, encoder, classifier, confidence_scorer

    print("Initializing Holmes AI models...")

    # Initialize preprocessor
    preprocessor = TransactionPreprocessor()
    print("✓ Preprocessor initialized")

    # Initialize confidence scorer
    try:
        confidence_scorer = ConfidenceScorer(
            taxonomy_path="src/config/taxonomy.json"
        )
        print("✓ Confidence scorer initialized")
    except Exception as e:
        print(f"Warning: Could not load taxonomy for confidence scorer: {e}")
        confidence_scorer = ConfidenceScorer()

    # Note: Encoder and classifier would be loaded from saved models
    # For now, they are initialized but not loaded
    print("Note: Encoder and classifier need to be trained and loaded")
    print("Holmes AI API ready!")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Holmes AI",
        "version": "1.0.0",
        "description": "AI-native transaction categorization engine",
        "endpoints": {
            "health": "/health",
            "categorize": "/api/v1/categorize",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_loaded = all([
        preprocessor is not None,
        confidence_scorer is not None
    ])

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        timestamp=datetime.now(),
        models_loaded=models_loaded
    )


@app.post("/api/v1/categorize", response_model=CategorizeResponse)
async def categorize_transactions(request: CategorizeRequest):
    """
    Categorize transactions into hierarchical categories.

    Args:
        request: CategorizeRequest with list of transactions

    Returns:
        CategorizeResponse with categorized transactions and metadata
    """
    start_time = time.time()

    if preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Models not initialized."
        )

    try:
        # Convert TransactionInput to dictionaries for processing
        transactions = [txn.model_dump() for txn in request.transactions]

        # Preprocess transactions
        preprocessed = preprocessor.preprocess_batch(transactions)

        # TODO: Encode with Sentence-BERT (requires trained model)
        # embeddings = encoder.encode_transactions(preprocessed)

        # TODO: Classify with LightGBM (requires trained model)
        # predictions = classifier.predict(embeddings)

        # For now, return mock predictions
        results = []
        confidences = []

        for i, txn in enumerate(preprocessed):
            # Mock prediction (would come from classifier)
            mock_category = CategoryPrediction(
                l1="Miscellaneous",
                l1_id="MSC",
                l2="Miscellaneous - Uncategorized",
                l2_id="MSC-UNC",
                l3="Miscellaneous - Uncategorized - Other",
                l3_id="MSC-UNC-OTH"
            )

            # Mock confidence (would come from confidence scorer)
            mock_confidence = 0.50

            processing_time = (time.time() - start_time) * 1000 / len(transactions)

            result = TransactionOutput(
                transaction_id=txn['transaction_id'],
                category=mock_category,
                confidence=mock_confidence,
                processing_time_ms=processing_time,
                explanation={
                    "note": "Models not trained yet. This is a mock prediction.",
                    "merchant_cleaned": txn.get('merchant_cleaned', ''),
                    "features": {
                        "spend_band": txn.get('spend_band', ''),
                        "temporal_pattern": txn.get('temporal_pattern', '')
                    }
                }
            )

            results.append(result)
            confidences.append(mock_confidence)

        # Calculate metadata
        low_confidence_count = sum(1 for c in confidences if c < 0.70)

        metadata = {
            "total_processed": len(results),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "low_confidence_count": low_confidence_count,
            "total_time_ms": (time.time() - start_time) * 1000
        }

        return CategorizeResponse(
            results=results,
            metadata=metadata
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing transactions: {str(e)}"
        )


@app.get("/api/v1/taxonomy")
async def get_taxonomy():
    """Get the full category taxonomy."""
    try:
        import json
        with open("src/config/taxonomy.json", 'r') as f:
            taxonomy = json.load(f)
        return taxonomy
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading taxonomy: {str(e)}"
        )


@app.get("/api/v1/stats")
async def get_stats():
    """Get API statistics."""
    return {
        "models": {
            "preprocessor": preprocessor is not None,
            "encoder": encoder is not None,
            "classifier": classifier is not None,
            "confidence_scorer": confidence_scorer is not None
        },
        "taxonomy": {
            "l1_categories": 15,
            "total_l3_categories": 45
        },
        "performance": {
            "target_latency_ms": 200,
            "target_f1_score": 0.90
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
