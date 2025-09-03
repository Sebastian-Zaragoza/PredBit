import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

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
    random_forest_classifier = RandomForestClassifier(n_estimators=330, max_features=1.0, max_leaf_nodes=8, min_samples_split=6, random_state=42, n_jobs=-1)
    random_forest_classifier.fit(X_train, y_train)
    umbral_ideal = 0.5316
    new_instance = random_forest_classifier.predict_proba(X_test)
    prediction_final = (new_instance[:, 1]>=umbral_ideal).astype(int)
    print(prediction_final)

if __name__ == "__main__":
    main()
