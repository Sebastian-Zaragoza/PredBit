import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_target_and_features(data_base: pd.DataFrame, config: dict)-> tuple[pd.DataFrame, pd.Series]:
    data_base['digital_wellbeing_score'] = (
            (data_base['anxiety_level'] <= 9.0)
    ).astype(int)
    data_base['digital_wellbeing_score'] = data_base['digital_wellbeing_score'] *10/3

    target_config = config["target"]
    target_name = target_config["name"]
    rule_column = target_config["rule"]["column"]
    threshold = target_config["rule"]["threshold"]
    data_base[target_name] = (data_base[rule_column]>=threshold).astype(int)

    scaler = StandardScaler()
    features = [f for f in config["features"]["numeric"] if f not in ['focus_score', 'anxiety_level', 'mod_score', 'sleep_hours']]
    X = data_base[features]
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=data_base.index)
    y = data_base[target_name]
    return X_scaled, y