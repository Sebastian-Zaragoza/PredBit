import sys
from pathlib import Path
import pandas as pd
import joblib

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))
from src.utils.config import load_config

def main():
    config = load_config()
    if config is None:
        return

    model_path = root_path / config["paths"]["model_dir"] / "random_forest_anxiety_level_pipeline.pkl"
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print("Error: Pipeline not found.")
        return

    new_data = {
        'daily_screen_time_min': [150],
        'notification_count': [120],
        'social_media_time_min': [90]
    }

    new_df = pd.DataFrame(new_data)
    prediction = pipeline.predict(new_df)

    if prediction[0] == 1:
        print("\nPrediction: Low Anxiety Levels")
    else:
        print("\nPrediction: High Anxiety Levels")

if __name__ == "__main__":
    main()