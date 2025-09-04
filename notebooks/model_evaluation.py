import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from src.data.load_data import load_config

def main():
    config = load_config(root_path)
    if config is None:
        return
    train_csv = pd.read_csv(root_path/config["paths"]["processed_dir"]/"train.csv")
    X_train = train_csv[config["features"]["numeric"]]
    y_train = train_csv[config["target"]["name"]]

    test_csv = pd.read_csv(root_path/config["paths"]["processed_dir"]/"test.csv")
    X_test = test_csv[config["features"]["numeric"]]
    y_test = test_csv[config["target"]["name"]]

    model_classifier = RandomForestClassifier( n_estimators=500,
    max_depth=None,
    max_leaf_nodes=16,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42)

    model_classifier.fit(X_train, y_train)
    predictions = model_classifier.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()
