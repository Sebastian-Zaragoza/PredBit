import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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

    """
    #Model
    param_grid = [
        {'n_estimators': [100, 200, 300], 'max_leaf_nodes': [10, 16, 20]},
        {'bootstrap': [False], 'n_estimators': [100, 200, 300], 'max_leaf_nodes': [10, 16, 20]},
    ]

    random_forest_classifier = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(random_forest_classifier, param_grid, cv=6, scoring='accuracy', return_train_score=True)
    grid_search.fit(X_train, y_train)

    print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor puntuación de validación cruzada: {grid_search.best_score_:.4f}")

    # 4. Evaluar el mejor modelo en el conjunto de prueba
    final_model = grid_search.best_estimator_
    y_pred = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nPrecisión final del modelo en el conjunto de prueba: {final_accuracy:.4f}")
    print(f"\nReporte de clasificación final:\n{classification_report(y_test, y_pred)}")
    
    """
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
