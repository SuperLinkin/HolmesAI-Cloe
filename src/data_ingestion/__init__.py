"""
Data ingestion package for Holmes AI.
"""

from .ingestion import DataIngestion
from .schema import (
    TransactionInput,
    TransactionNormalized,
    CategoryPrediction,
    TransactionOutput
)

__all__ = [
    'DataIngestion',
    'TransactionInput',
    'TransactionNormalized',
    'CategoryPrediction',
    'TransactionOutput'
]
