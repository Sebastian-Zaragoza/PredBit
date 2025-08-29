import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

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

    random_forest_classifier = RandomForestClassifier(n_estimators=10, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    """
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    """
    """
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"\nClassification report: {classification_report(y_test, y_pred)}")
    print(f"\nConfusion matrix: \n{confusion_matrix(y_test, y_pred)}")
    """

    #Confuse Matrix
    y_keep_habit = (y_train == 1)
    y_train_pred = cross_val_predict(
        random_forest_classifier, X_train, y_keep_habit, cv=3
    )
    confusion_matrix_results = confusion_matrix(y_keep_habit, y_train_pred)
    print(f"\nConfusion matrix results: \n{confusion_matrix_results}")

    #Decision Function
    y_scores = cross_val_predict(random_forest_classifier, X_train, y_keep_habit, cv=3, n_jobs=-1, method="predict_proba")
    precisions, recalls, thresholds = precision_recall_curve(y_keep_habit, y_scores)
    plt.plot(thresholds, precisions[:-1], "b--", label="precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g--", label="recall", linewidth=2)
    plt.axvline(x=0, color="k", linewidth=2)
    plt.show()



if __name__ == "__main__":
    main()
