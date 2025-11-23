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
from ..feedback.feedback_storage import FeedbackStorage

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


class FeedbackRequest(BaseModel):
    """Request model for feedback submission."""
    merchant: str = Field(..., description="Merchant name")
    amount: float = Field(..., gt=0, description="Transaction amount")
    date: str = Field(..., description="Transaction date (YYYY-MM-DD)")
    mcc_code: Optional[str] = Field(None, description="MCC code")
    transaction_id: Optional[str] = Field(None, description="Transaction ID")
    predicted_l1: str = Field(..., description="Predicted L1 category")
    predicted_l2: str = Field(..., description="Predicted L2 category")
    predicted_l3: str = Field(..., description="Predicted L3 category")
    predicted_confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    corrected_l1: str = Field(..., description="Corrected L1 category")
    corrected_l2: str = Field(..., description="Corrected L2 category")
    corrected_l3: str = Field(..., description="Corrected L3 category")
    user_id: Optional[str] = Field(None, description="User ID")
    notes: Optional[str] = Field(None, description="Additional notes")

    class Config:
        json_schema_extra = {
            "example": {
                "merchant": "STARBUCKS #4532",
                "amount": 5.25,
                "date": "2025-01-15",
                "mcc_code": "5812",
                "predicted_l1": "Shopping",
                "predicted_l2": "Retail",
                "predicted_l3": "Other",
                "predicted_confidence": 0.65,
                "corrected_l1": "Dining",
                "corrected_l2": "Coffee Shops",
                "corrected_l3": "Starbucks",
                "notes": "Misclassified coffee shop as retail"
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_id: int
    message: str
    total_feedback: int


# Global model instances (loaded on startup)
preprocessor: Optional[TransactionPreprocessor] = None
encoder: Optional[SentenceBERTEncoder] = None
classifier: Optional[LightGBMClassifier] = None
confidence_scorer: Optional[ConfidenceScorer] = None
feedback_storage: Optional[FeedbackStorage] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on application startup."""
    global preprocessor, encoder, classifier, confidence_scorer, feedback_storage

    print("Initializing Holmes AI models...")

    # Initialize feedback storage
    try:
        feedback_storage = FeedbackStorage(db_path="data/feedback.db")
        print("[OK] Feedback storage initialized")
    except Exception as e:
        print(f"Warning: Could not initialize feedback storage: {e}")
        feedback_storage = None

    # Initialize preprocessor
    preprocessor = TransactionPreprocessor()
    print("[OK] Preprocessor initialized")

    # Initialize confidence scorer
    try:
        confidence_scorer = ConfidenceScorer(
            taxonomy_path="src/config/taxonomy.json"
        )
        print("[OK] Confidence scorer initialized")
    except Exception as e:
        print(f"Warning: Could not load taxonomy for confidence scorer: {e}")
        confidence_scorer = ConfidenceScorer()

    # Try to load trained models if they exist
    from pathlib import Path
    encoder_path = Path("data/models/sentence_bert")
    classifier_path = Path("data/models/lightgbm")

    if encoder_path.exists() and classifier_path.exists():
        try:
            print("Loading trained models...")

            # Load Sentence-BERT encoder
            encoder = SentenceBERTEncoder(model_path=str(encoder_path))
            print("[OK] Sentence-BERT encoder loaded")

            # Load LightGBM classifier
            classifier = LightGBMClassifier(taxonomy_path="src/config/taxonomy.json")
            classifier.load(str(classifier_path))
            print("[OK] LightGBM classifiers loaded")

            print("[OK] All trained models loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load trained models: {e}")
            print("API will use fallback mode")
            encoder = None
            classifier = None
    else:
        print("Note: Trained models not found. API will use fallback mode")
        encoder = None
        classifier = None

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
            "confidence_scorer": confidence_scorer is not None,
            "feedback_storage": feedback_storage is not None
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


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a prediction correction.

    This endpoint allows users to correct misclassified transactions,
    which will be stored for future model retraining.

    Args:
        request: FeedbackRequest with prediction and correction

    Returns:
        FeedbackResponse with feedback ID and summary
    """
    if feedback_storage is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback storage not available"
        )

    try:
        feedback_id = feedback_storage.add_feedback(
            merchant=request.merchant,
            amount=request.amount,
            date=request.date,
            mcc_code=request.mcc_code,
            transaction_id=request.transaction_id,
            predicted_l1=request.predicted_l1,
            predicted_l2=request.predicted_l2,
            predicted_l3=request.predicted_l3,
            predicted_confidence=request.predicted_confidence,
            corrected_l1=request.corrected_l1,
            corrected_l2=request.corrected_l2,
            corrected_l3=request.corrected_l3,
            user_id=request.user_id,
            feedback_type="correction",
            notes=request.notes
        )

        total_feedback = feedback_storage.get_feedback_count()

        return FeedbackResponse(
            feedback_id=feedback_id,
            message="Feedback submitted successfully. Thank you for helping improve the model!",
            total_feedback=total_feedback
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error storing feedback: {str(e)}"
        )


@app.get("/api/v1/feedback/stats")
async def get_feedback_stats():
    """
    Get feedback statistics and summary.

    Returns:
        Dictionary with feedback counts and statistics
    """
    if feedback_storage is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback storage not available"
        )

    try:
        summary = feedback_storage.get_feedback_summary()
        return {
            "summary": summary,
            "status": "ok"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving feedback stats: {str(e)}"
        )


@app.get("/api/v1/feedback/patterns")
async def get_misclassification_patterns(min_count: int = 3):
    """
    Get common misclassification patterns from user feedback.

    Args:
        min_count: Minimum occurrences to be considered a pattern

    Returns:
        List of common misclassification patterns
    """
    if feedback_storage is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback storage not available"
        )

    try:
        patterns_df = feedback_storage.get_misclassification_patterns(min_count=min_count)
        patterns = patterns_df.to_dict('records') if not patterns_df.empty else []

        return {
            "patterns": patterns,
            "count": len(patterns),
            "min_count_threshold": min_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving patterns: {str(e)}"
        )


@app.get("/api/v1/feedback/export")
async def export_feedback(unused_only: bool = True):
    """
    Export feedback data for analysis.

    Args:
        unused_only: If True, only export feedback not yet used in training

    Returns:
        CSV data of feedback entries
    """
    if feedback_storage is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback storage not available"
        )

    try:
        import tempfile
        from fastapi.responses import FileResponse

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name

        # Export to CSV
        count = feedback_storage.export_feedback_csv(temp_path, unused_only=unused_only)

        if count == 0:
            raise HTTPException(
                status_code=404,
                detail="No feedback data available for export"
            )

        return FileResponse(
            path=temp_path,
            filename=f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            media_type="text/csv"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting feedback: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
