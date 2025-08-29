import sys
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import seaborn as sns

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

    print(full_data.head())
    print(full_data.info())

    sleep_hours_anxiety_level = pd.concat([full_data["sleep_hours"], full_data["anxiety_level"]], axis=1)
    sns.pairplot(data=sleep_hours_anxiety_level)
    plt.show()

if __name__ == "__main__":
    main()
