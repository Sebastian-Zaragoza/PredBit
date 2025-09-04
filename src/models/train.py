from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))
from src.utils.config import load_config

def main():
    config = load_config()
    if config is None:
        print("Error: The configuration file was not found.")
        return
    train_csv = pd.read_csv(root_path/config['paths']['processed_dir']/'train.csv')
    X_train = train_csv[config['features']['numeric']]
    y_train = train_csv[config['target']['name']]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=500, max_depth=None, max_leaf_nodes=16, class_weight="balanced", n_jobs=-1, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    model_dir = root_path/config['paths']['model_dir']
    model_path = model_dir/"random_forest_anxiety_level_pipeline.pkl"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print("Model saved in {}".format(model_path))

if __name__ == "__main__":
    main()