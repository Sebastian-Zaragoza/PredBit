import pandas as pd

def create_target_and_features(data_base: pd.DataFrame, config: dict)-> tuple[pd.DataFrame, pd.Series]:
    target_config = config["target"]
    target_name = target_config["name"]
    rule_column = target_config["rule"]["column"]
    threshold = target_config["rule"]["threshold"]
    data_base[target_name] = (data_base[rule_column]>=threshold).astype(int)

    features = config["features"]["numeric"]
    X = data_base[features]
    y = data_base["target_name"]
    return X, y