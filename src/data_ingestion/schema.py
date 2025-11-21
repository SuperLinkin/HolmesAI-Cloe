"""
Data schema definitions for transaction ingestion.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """Input schema for raw transaction data."""

    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    merchant_raw: str = Field(..., description="Raw merchant description from bank feed")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    channel: Optional[str] = Field(default=None, description="Payment channel: online, pos, atm")
    location: Optional[str] = Field(default=None, description="Merchant location")
    mcc_code: Optional[str] = Field(default=None, description="Merchant Category Code")

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        if v == 0:
            raise ValueError("Amount cannot be zero")
        return v

    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v):
        # Basic ISO 4217 validation
        if len(v) != 3:
            raise ValueError("Currency must be 3-letter ISO 4217 code")
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_20240115_001",
                "merchant_raw": "AMZN MKTP US*2A3B4C5D6",
                "amount": 49.99,
                "currency": "USD",
                "timestamp": "2024-01-15T14:32:00Z",
                "channel": "online",
                "location": "Seattle, WA",
                "mcc_code": "5942"
            }
        }


class TransactionNormalized(BaseModel):
    """Normalized transaction schema after preprocessing."""

    transaction_id: str
    merchant_raw: str
    merchant_cleaned: str
    amount: float
    currency: str
    timestamp: datetime
    channel: Optional[str]
    location: Optional[str]
    mcc_code: Optional[str]

    # Derived fields
    day_of_week: int  # 0=Monday, 6=Sunday
    hour_of_day: int  # 0-23
    month: int  # 1-12

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_20240115_001",
                "merchant_raw": "AMZN MKTP US*2A3B4C5D6",
                "merchant_cleaned": "amazon marketplace",
                "amount": 49.99,
                "currency": "USD",
                "timestamp": "2024-01-15T14:32:00Z",
                "channel": "online",
                "location": "Seattle, WA",
                "mcc_code": "5942",
                "day_of_week": 0,
                "hour_of_day": 14,
                "month": 1
            }
        }


class CategoryPrediction(BaseModel):
    """Predicted category with confidence scores."""

    l1: str = Field(..., description="Level 1 category")
    l1_id: str = Field(..., description="Level 1 category ID")
    l2: str = Field(..., description="Level 2 category")
    l2_id: str = Field(..., description="Level 2 category ID")
    l3: str = Field(..., description="Level 3 category")
    l3_id: str = Field(..., description="Level 3 category ID")

    class Config:
        json_schema_extra = {
            "example": {
                "l1": "Shopping",
                "l1_id": "SHP",
                "l2": "Shopping - Online",
                "l2_id": "SHP-ONL",
                "l3": "Shopping - Online - Amazon",
                "l3_id": "SHP-ONL-AMZ"
            }
        }


class TransactionOutput(BaseModel):
    """Complete transaction output with categorization."""

    transaction_id: str
    category: CategoryPrediction
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    processing_time_ms: float
    explanation: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_20240115_001",
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
                "explanation": {
                    "top_features": [],
                    "similar_transactions": []
                }
            }
        }
