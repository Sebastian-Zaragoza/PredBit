import sys
from pathlib import Path
import pandas as pd

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.data.load_data import load_config, load_raw_data
from src.data.preprocess import create_target_and_features

def main():
    config = load_config(root_path)
    if config is None:
        return
    database = load_raw_data(root_path, config)
    if database is None:
        return
    X, y =  create_target_and_features(database, config)
    full_data = pd.concat([X,y], axis=1)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(full_data.head())
    print(full_data.info())
    print(full_data.corr(numeric_only=True))

    """
    Relations:
    daily_screen_time_min -> mood_score -> 0.07
    sleep_hours -> digital_wellbeing_score -> 0.23
    notification_count -> anxiety_level -> 0.30
    social_media_time_min -> anxiety_level -> 0.31
    focus_score -> digital_wellbeing_score -> 0.23
    mood_score -> digital_wellbeing_score -> 0.08
    anxiety_level -> notification_account -> 0.3
    anxiety_level -> social_media_time_min -> 0.3
    digital_wellbeing_score -> focus_score -> 0.23
    digital_wellbeing_score -> sleep_hours -> 0.24
    """
if __name__ == "__main__":
    main()
