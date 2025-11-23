"""
Feedback Storage Module for Holmes AI

Stores user corrections and feedback for model improvement and retraining.
Uses SQLite for lightweight, serverless storage.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


class FeedbackStorage:
    """Manages storage and retrieval of user feedback for model improvement."""

    def __init__(self, db_path: str = "data/feedback.db"):
        """
        Initialize feedback storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT,
                    merchant TEXT NOT NULL,
                    amount REAL NOT NULL,
                    date TEXT NOT NULL,
                    mcc_code TEXT,
                    predicted_l1 TEXT NOT NULL,
                    predicted_l2 TEXT NOT NULL,
                    predicted_l3 TEXT NOT NULL,
                    predicted_confidence REAL NOT NULL,
                    corrected_l1 TEXT NOT NULL,
                    corrected_l2 TEXT NOT NULL,
                    corrected_l3 TEXT NOT NULL,
                    user_id TEXT,
                    feedback_type TEXT DEFAULT 'correction',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    used_in_training INTEGER DEFAULT 0,
                    training_run_id TEXT
                )
            """)

            # Retraining history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retraining_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    feedback_count INTEGER NOT NULL,
                    training_samples INTEGER NOT NULL,
                    validation_samples INTEGER NOT NULL,
                    l1_accuracy REAL,
                    l2_accuracy REAL,
                    l3_accuracy REAL,
                    l1_f1 REAL,
                    l2_f1 REAL,
                    l3_f1 REAL,
                    training_duration_sec REAL,
                    model_path TEXT,
                    config TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT DEFAULT 'started'
                )
            """)

            # Feedback statistics view
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS feedback_stats AS
                SELECT
                    feedback_type,
                    predicted_l1,
                    corrected_l1,
                    COUNT(*) as count,
                    AVG(predicted_confidence) as avg_confidence,
                    MIN(created_at) as first_feedback,
                    MAX(created_at) as latest_feedback
                FROM feedback
                GROUP BY feedback_type, predicted_l1, corrected_l1
            """)

            conn.commit()
            print(f"[OK] Feedback database initialized at {self.db_path}")

    def add_feedback(
        self,
        merchant: str,
        amount: float,
        date: str,
        predicted_l1: str,
        predicted_l2: str,
        predicted_l3: str,
        predicted_confidence: float,
        corrected_l1: str,
        corrected_l2: str,
        corrected_l3: str,
        transaction_id: Optional[str] = None,
        mcc_code: Optional[str] = None,
        user_id: Optional[str] = None,
        feedback_type: str = "correction",
        notes: Optional[str] = None
    ) -> int:
        """
        Add user feedback to database.

        Args:
            merchant: Merchant name
            amount: Transaction amount
            date: Transaction date (YYYY-MM-DD)
            predicted_l1: Predicted L1 category
            predicted_l2: Predicted L2 category
            predicted_l3: Predicted L3 category
            predicted_confidence: Model confidence (0-1)
            corrected_l1: User-corrected L1 category
            corrected_l2: User-corrected L2 category
            corrected_l3: User-corrected L3 category
            transaction_id: Optional transaction identifier
            mcc_code: Optional MCC code
            user_id: Optional user identifier
            feedback_type: Type of feedback ('correction', 'validation', 'report')
            notes: Optional user notes

        Returns:
            Feedback ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (
                    transaction_id, merchant, amount, date, mcc_code,
                    predicted_l1, predicted_l2, predicted_l3, predicted_confidence,
                    corrected_l1, corrected_l2, corrected_l3,
                    user_id, feedback_type, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction_id, merchant, amount, date, mcc_code,
                predicted_l1, predicted_l2, predicted_l3, predicted_confidence,
                corrected_l1, corrected_l2, corrected_l3,
                user_id, feedback_type, notes, datetime.now().isoformat()
            ))
            conn.commit()
            feedback_id = cursor.lastrowid
            print(f"[OK] Feedback stored (ID: {feedback_id})")
            return feedback_id

    def get_feedback_count(self, unused_only: bool = False) -> int:
        """
        Get count of feedback entries.

        Args:
            unused_only: If True, only count feedback not yet used in training

        Returns:
            Count of feedback entries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if unused_only:
                cursor.execute("SELECT COUNT(*) FROM feedback WHERE used_in_training = 0")
            else:
                cursor.execute("SELECT COUNT(*) FROM feedback")
            return cursor.fetchone()[0]

    def get_feedback_for_training(
        self,
        min_samples: int = 100,
        unused_only: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve feedback data for model retraining.

        Args:
            min_samples: Minimum number of samples to retrieve
            unused_only: If True, only get feedback not yet used in training

        Returns:
            DataFrame with feedback data formatted for training
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT
                    merchant,
                    amount,
                    date,
                    mcc_code,
                    corrected_l1 as L1,
                    corrected_l2 as L2,
                    corrected_l3 as L3,
                    id as feedback_id
                FROM feedback
            """
            if unused_only:
                query += " WHERE used_in_training = 0"

            df = pd.read_sql_query(query, conn)

            if len(df) < min_samples:
                print(f"[WARN] Only {len(df)} feedback samples available (min: {min_samples})")

            return df

    def mark_feedback_as_used(
        self,
        feedback_ids: List[int],
        training_run_id: str
    ):
        """
        Mark feedback entries as used in training.

        Args:
            feedback_ids: List of feedback IDs used in training
            training_run_id: Identifier for the training run
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(feedback_ids))
            cursor.execute(f"""
                UPDATE feedback
                SET used_in_training = 1, training_run_id = ?
                WHERE id IN ({placeholders})
            """, [training_run_id] + feedback_ids)
            conn.commit()
            print(f"[OK] Marked {len(feedback_ids)} feedback entries as used (run: {training_run_id})")

    def add_retraining_run(
        self,
        run_id: str,
        feedback_count: int,
        training_samples: int,
        validation_samples: int,
        config: Optional[Dict] = None
    ) -> int:
        """
        Record start of a retraining run.

        Args:
            run_id: Unique identifier for this run
            feedback_count: Number of feedback samples used
            training_samples: Total training samples
            validation_samples: Total validation samples
            config: Training configuration

        Returns:
            Retraining run database ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO retraining_history (
                    run_id, feedback_count, training_samples, validation_samples,
                    config, started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, feedback_count, training_samples, validation_samples,
                json.dumps(config or {}), datetime.now().isoformat(), 'started'
            ))
            conn.commit()
            return cursor.lastrowid

    def update_retraining_run(
        self,
        run_id: str,
        l1_accuracy: Optional[float] = None,
        l2_accuracy: Optional[float] = None,
        l3_accuracy: Optional[float] = None,
        l1_f1: Optional[float] = None,
        l2_f1: Optional[float] = None,
        l3_f1: Optional[float] = None,
        training_duration_sec: Optional[float] = None,
        model_path: Optional[str] = None,
        status: str = 'completed'
    ):
        """
        Update retraining run with results.

        Args:
            run_id: Training run identifier
            l1_accuracy: L1 accuracy
            l2_accuracy: L2 accuracy
            l3_accuracy: L3 accuracy
            l1_f1: L1 macro F1 score
            l2_f1: L2 macro F1 score
            l3_f1: L3 macro F1 score
            training_duration_sec: Training duration in seconds
            model_path: Path to saved model
            status: Status ('started', 'completed', 'failed')
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE retraining_history
                SET l1_accuracy = ?, l2_accuracy = ?, l3_accuracy = ?,
                    l1_f1 = ?, l2_f1 = ?, l3_f1 = ?,
                    training_duration_sec = ?, model_path = ?,
                    completed_at = ?, status = ?
                WHERE run_id = ?
            """, (
                l1_accuracy, l2_accuracy, l3_accuracy,
                l1_f1, l2_f1, l3_f1,
                training_duration_sec, model_path,
                datetime.now().isoformat(), status, run_id
            ))
            conn.commit()
            print(f"[OK] Updated retraining run {run_id} (status: {status})")

    def get_retraining_history(self, limit: int = 10) -> pd.DataFrame:
        """
        Get history of retraining runs.

        Args:
            limit: Maximum number of runs to retrieve

        Returns:
            DataFrame with retraining history
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"""
                SELECT * FROM retraining_history
                ORDER BY started_at DESC
                LIMIT {limit}
            """, conn)

    def get_feedback_statistics(self) -> pd.DataFrame:
        """
        Get aggregated feedback statistics.

        Returns:
            DataFrame with feedback stats grouped by category transitions
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM feedback_stats", conn)

    def get_misclassification_patterns(self, min_count: int = 3) -> pd.DataFrame:
        """
        Identify common misclassification patterns.

        Args:
            min_count: Minimum occurrences to be considered a pattern

        Returns:
            DataFrame with misclassification patterns
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"""
                SELECT
                    predicted_l1,
                    predicted_l2,
                    predicted_l3,
                    corrected_l1,
                    corrected_l2,
                    corrected_l3,
                    COUNT(*) as occurrence_count,
                    AVG(predicted_confidence) as avg_confidence,
                    GROUP_CONCAT(merchant, ', ') as sample_merchants
                FROM feedback
                WHERE predicted_l3 != corrected_l3
                GROUP BY predicted_l1, predicted_l2, predicted_l3, corrected_l1, corrected_l2, corrected_l3
                HAVING COUNT(*) >= {min_count}
                ORDER BY COUNT(*) DESC
            """, conn)

    def export_feedback_csv(self, output_path: str, unused_only: bool = False) -> int:
        """
        Export feedback data to CSV file.

        Args:
            output_path: Path to output CSV file
            unused_only: If True, only export unused feedback

        Returns:
            Number of rows exported
        """
        df = self.get_feedback_for_training(min_samples=0, unused_only=unused_only)
        df.to_csv(output_path, index=False)
        print(f"[OK] Exported {len(df)} feedback entries to {output_path}")
        return len(df)

    def get_feedback_summary(self) -> Dict:
        """
        Get summary statistics of all feedback.

        Returns:
            Dictionary with summary statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total counts
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feedback WHERE used_in_training = 1")
            used_feedback = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feedback WHERE predicted_l3 != corrected_l3")
            corrections = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT predicted_l1) FROM feedback")
            affected_l1 = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM retraining_history WHERE status = 'completed'")
            retraining_runs = cursor.fetchone()[0]

            return {
                "total_feedback": total_feedback,
                "used_in_training": used_feedback,
                "unused_feedback": total_feedback - used_feedback,
                "corrections": corrections,
                "validations": total_feedback - corrections,
                "affected_l1_categories": affected_l1,
                "retraining_runs": retraining_runs
            }


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feedback Storage CLI")
    parser.add_argument('--db-path', default='data/feedback.db', help='Database path')
    parser.add_argument('--action', choices=['init', 'stats', 'export', 'list'], required=True)
    parser.add_argument('--output', help='Output path for export')

    args = parser.parse_args()

    storage = FeedbackStorage(db_path=args.db_path)

    if args.action == 'init':
        print("Database initialized successfully")

    elif args.action == 'stats':
        summary = storage.get_feedback_summary()
        print("\n=== Feedback Summary ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n=== Feedback Statistics ===")
        stats = storage.get_feedback_statistics()
        if not stats.empty:
            print(stats.to_string(index=False))
        else:
            print("  No feedback data yet")

    elif args.action == 'export':
        if not args.output:
            print("Error: --output required for export action")
        else:
            count = storage.export_feedback_csv(args.output)
            print(f"Exported {count} entries")

    elif args.action == 'list':
        print("\n=== Recent Retraining Runs ===")
        history = storage.get_retraining_history(limit=5)
        if not history.empty:
            print(history.to_string(index=False))
        else:
            print("  No retraining runs yet")
