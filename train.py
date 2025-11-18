import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def load_olivetti_data():
    from sklearn.datasets import fetch_olivetti_faces
    print("Loading Olivetti faces dataset...")
    olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
    return olivetti.data, olivetti.target

def main():
    X, y = load_olivetti_data()
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y))}")
    
    print("\nSplitting data: 70% train, 30% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=42,
        stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining DecisionTreeClassifier with optimized parameters...")
    model = DecisionTreeClassifier(
        random_state=42,
        criterion='gini',          # Gini impurity
        splitter='best',           # Best split at each node
        max_depth=30,              # Increased depth for better training accuracy
        min_samples_split=2,       # Minimum samples to split a node
        min_samples_leaf=1,        # Minimum samples in leaf node
        max_features='sqrt',       # Use sqrt of features for each split
        min_impurity_decrease=0.0  # No minimum impurity decrease required
    )

    model.fit(X_train_scaled, y_train)

    # ✅ Only training accuracy is computed & printed
    train_accuracy = model.score(X_train_scaled, y_train)
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

    # Save model + scaler
    model_filename = 'savedmodel.pth'
    print(f"\nSaving model to {model_filename}...")
    model_package = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_package, model_filename)
    print("Model and scaler saved successfully!")

    # Optional: save test data for separate evaluation in test.py
    test_data = {
        'X_test': X_test_scaled,
        'y_test': y_test
    }
    joblib.dump(test_data, 'test_data.pkl')
    print("Test data saved for evaluation.")

    print(f"\nModel Details:")
    print(f"  - Max depth reached: {model.get_depth()}")
    print(f"  - Number of leaves: {model.get_n_leaves()}")
    print(f"  - Number of features used: {model.n_features_in_}")
    print(f"  - Feature importance (top 5): {np.argsort(model.feature_importances_)[-5:]}")

if __name__ == "__main__":
    main()