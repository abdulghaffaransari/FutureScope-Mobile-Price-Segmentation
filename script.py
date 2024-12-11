
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse
import os
import pandas as pd


def model_fn(model_dir):
    """Load and return the trained model."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    print("[INFO] Extracting arguments...")
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")
    args, _ = parser.parse_known_args()

    print(f"SKLearn Version: {joblib.__version__}")
    print("[INFO] Reading data...")

    # Load datasets
    train_path = os.path.join(args.train, args.train_file)
    test_path = os.path.join(args.test, args.test_file)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Extract features and labels
    features = train_df.columns[:-1]
    label = train_df.columns[-1]
    X_train, y_train = train_df[features], train_df[label]
    X_test, y_test = test_df[features], test_df[label]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    print("Training RandomForest model...")

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.random_state, verbose=1
    )
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    # Evaluate model
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)

    print(f"Test Accuracy: {test_acc}")
    print("Classification Report:\n", test_report)
