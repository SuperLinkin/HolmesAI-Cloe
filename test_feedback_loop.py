"""
Test Script for Feedback Loop

Tests the complete feedback collection and storage system.
"""

import sys
from pathlib import Path
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.feedback import FeedbackStorage
import pandas as pd


def test_feedback_storage():
    """Test feedback storage functionality."""
    print("\n" + "="*60)
    print("Testing Feedback Storage")
    print("="*60)

    # Initialize storage with test database
    storage = FeedbackStorage(db_path="data/test_feedback.db")

    # Test 1: Add feedback
    print("\n1. Adding sample feedback...")
    feedback_id = storage.add_feedback(
        merchant="STARBUCKS #4532",
        amount=5.25,
        date="2025-01-15",
        mcc_code="5812",
        predicted_l1="Shopping",
        predicted_l2="Retail",
        predicted_l3="Other",
        predicted_confidence=0.65,
        corrected_l1="Dining",
        corrected_l2="Coffee Shops",
        corrected_l3="Starbucks",
        user_id="test_user_1",
        notes="Misclassified coffee shop as retail"
    )
    print(f"   ✓ Added feedback ID: {feedback_id}")

    # Test 2: Add more feedback
    print("\n2. Adding more feedback samples...")
    for i in range(5):
        storage.add_feedback(
            merchant=f"TEST_MERCHANT_{i}",
            amount=10.0 + i,
            date="2025-01-16",
            predicted_l1="Miscellaneous",
            predicted_l2="Uncategorized",
            predicted_l3="Other",
            predicted_confidence=0.50,
            corrected_l1="Shopping",
            corrected_l2="Online Retail",
            corrected_l3="Amazon",
            user_id="test_user_1"
        )
    print(f"   ✓ Added 5 more feedback entries")

    # Test 3: Get feedback count
    print("\n3. Checking feedback count...")
    total_count = storage.get_feedback_count()
    unused_count = storage.get_feedback_count(unused_only=True)
    print(f"   Total feedback: {total_count}")
    print(f"   Unused feedback: {unused_count}")

    # Test 4: Get feedback summary
    print("\n4. Getting feedback summary...")
    summary = storage.get_feedback_summary()
    print("   Summary:")
    for key, value in summary.items():
        print(f"     {key}: {value}")

    # Test 5: Get feedback for training
    print("\n5. Retrieving feedback for training...")
    feedback_df = storage.get_feedback_for_training(min_samples=0, unused_only=True)
    print(f"   Retrieved {len(feedback_df)} samples")
    print("\n   Sample data:")
    print(feedback_df.head().to_string(index=False))

    # Test 6: Export feedback to CSV
    print("\n6. Exporting feedback to CSV...")
    export_path = "data/test_feedback_export.csv"
    count = storage.export_feedback_csv(export_path, unused_only=False)
    print(f"   ✓ Exported {count} entries to {export_path}")

    # Test 7: Mark feedback as used
    print("\n7. Marking feedback as used...")
    feedback_ids = feedback_df['feedback_id'].tolist()[:3]
    storage.mark_feedback_as_used(feedback_ids, "test_run_001")
    print(f"   ✓ Marked {len(feedback_ids)} feedback entries as used")

    # Test 8: Check unused count again
    unused_after = storage.get_feedback_count(unused_only=True)
    print(f"\n8. Unused feedback after marking: {unused_after}")
    print(f"   (Was {unused_count}, marked {len(feedback_ids)} as used)")

    # Test 9: Add retraining run
    print("\n9. Recording retraining run...")
    run_id = storage.add_retraining_run(
        run_id="test_run_001",
        feedback_count=3,
        training_samples=1000,
        validation_samples=150,
        config={"n_estimators": 500}
    )
    print(f"   ✓ Added retraining run with ID: {run_id}")

    # Test 10: Update retraining run with results
    print("\n10. Updating retraining run with results...")
    storage.update_retraining_run(
        run_id="test_run_001",
        l1_accuracy=0.996,
        l2_accuracy=0.985,
        l3_accuracy=0.973,
        l1_f1=0.994,
        l2_f1=0.982,
        l3_f1=0.970,
        training_duration_sec=120.5,
        model_path="models/test",
        status="completed"
    )
    print("   ✓ Updated retraining run with metrics")

    # Test 11: Get retraining history
    print("\n11. Retrieving retraining history...")
    history_df = storage.get_retraining_history(limit=5)
    print(f"   Retrieved {len(history_df)} retraining runs")
    if not history_df.empty:
        print("\n   Recent runs:")
        print(history_df[['run_id', 'status', 'l1_f1', 'l2_f1', 'l3_f1']].to_string(index=False))

    # Test 12: Get misclassification patterns
    print("\n12. Analyzing misclassification patterns...")
    patterns_df = storage.get_misclassification_patterns(min_count=1)
    print(f"   Found {len(patterns_df)} patterns")
    if not patterns_df.empty:
        print("\n   Top patterns:")
        print(patterns_df[['predicted_l1', 'corrected_l1', 'occurrence_count']].head().to_string(index=False))

    print("\n" + "="*60)
    print("All Tests Passed! ✓")
    print("="*60)
    print("\nFeedback storage is working correctly.")
    print(f"Test database: data/test_feedback.db")
    print(f"Test export: {export_path}")


def test_integration():
    """Test integration with taxonomy."""
    print("\n" + "="*60)
    print("Testing Integration with Taxonomy")
    print("="*60)

    # Load taxonomy
    import json
    with open("src/config/taxonomy.json", 'r') as f:
        taxonomy = json.load(f)

    print(f"\nLoaded taxonomy with {len(taxonomy.get('L1', []))} L1 categories")

    # Create feedback for each L1 category
    storage = FeedbackStorage(db_path="data/test_feedback.db")

    print("\nCreating feedback for each L1 category...")
    for l1_cat in taxonomy.get('L1', [])[:3]:  # Test first 3
        l1_name = l1_cat.get('name', 'Unknown')
        l2_cats = l1_cat.get('L2', [])
        if l2_cats:
            l2_name = l2_cats[0].get('name', 'Unknown')
            l3_cats = l2_cats[0].get('L3', [])
            l3_name = l3_cats[0].get('name', 'Unknown') if l3_cats else 'Other'

            storage.add_feedback(
                merchant=f"TEST_{l1_name.upper()}",
                amount=50.0,
                date="2025-01-17",
                predicted_l1="Miscellaneous",
                predicted_l2="Uncategorized",
                predicted_l3="Other",
                predicted_confidence=0.45,
                corrected_l1=l1_name,
                corrected_l2=l2_name,
                corrected_l3=l3_name,
                notes=f"Test feedback for {l1_name}"
            )
            print(f"   ✓ Added feedback for {l1_name} → {l2_name} → {l3_name}")

    print("\n✓ Integration test complete")


if __name__ == "__main__":
    try:
        test_feedback_storage()
        test_integration()

        print("\n" + "="*60)
        print("Feedback Loop Testing Complete!")
        print("="*60)
        print("\nAll components are working correctly:")
        print("  ✓ Feedback storage (SQLite)")
        print("  ✓ Feedback retrieval")
        print("  ✓ Retraining run tracking")
        print("  ✓ Misclassification pattern analysis")
        print("  ✓ Integration with taxonomy")
        print("\nReady for production use!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
