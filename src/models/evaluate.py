import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))
from src.utils.config import load_config

def main():
    config = load_config()
    if config is None:
        return

    test_csv = pd.read_csv(root_path / config["paths"]["processed_dir"] / "test.csv")
    X_test = test_csv[config["features"]["numeric"]]
    y_test = test_csv[config["target"]["name"]]

    model_path = root_path / config["paths"]["model_dir"] / "random_forest_anxiety_level_pipeline.pkl"
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print("Error: Pipeline not found.")
        return

    predictions = pipeline.predict(X_test)
    print(f"\nModels Precision: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()