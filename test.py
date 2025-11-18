"""
Test script for evaluating the trained model
Loads the saved model package and computes test accuracy
"""

import joblib
import numpy as np


def main():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print("\nLoading saved model package...")
    # Load the saved model package (includes model and scaler)
    model_package = joblib.load("savedmodel.pth")

    # Your train.py saved a dict: {'model': model, 'scaler': scaler}
    model = model_package["model"]
    scaler = model_package.get("scaler", None)  # Not used here, but loaded for completeness

    print("✓ Model and scaler loaded successfully!")

    # Load test data
    print("\nLoading test data...")
    test_data = joblib.load("test_data.pkl")
    X_test = test_data["X_test"]  # already scaled in train.py
    y_test = test_data["y_test"]
    print(f"✓ Test set size: {X_test.shape[0]} samples")

    # Evaluate model
    print("\nEvaluating model on test set...")
    test_accuracy = model.score(X_test, y_test)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    correct_predictions = np.sum(y_pred == y_test)
    total_predictions = len(y_test)
    error_rate = (1 - test_accuracy) * 100

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:          {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Correct Predictions:    {correct_predictions}/{total_predictions}")
    print(f"Incorrect Predictions:  {total_predictions - correct_predictions}/{total_predictions}")
    print(f"Error Rate:             {error_rate:.2f}%")
    print("=" * 60)

    # Per-class accuracy analysis
    print("\nPer-class performance:")
    unique_classes = np.unique(y_test)
    class_accuracies = []

    for cls in unique_classes[:5]:  # Show first 5 classes as examples
        mask = y_test == cls
        if np.sum(mask) > 0:
            cls_acc = np.sum(y_pred[mask] == y_test[mask]) / np.sum(mask)
            class_accuracies.append(cls_acc)
            print(f"  Class {int(cls)}: {cls_acc:.2f} ({cls_acc * 100:.1f}%)")

    if class_accuracies:
        print(f"\nAverage per-class accuracy (first 5 classes): {np.mean(class_accuracies):.4f}")

    print("\n" + "=" * 60)
    print("✓ EVALUATION COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()