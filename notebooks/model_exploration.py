import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

    """
    test_csv = pd.read_csv(root_path/config["paths"]["processed_dir"]/"test.csv")
    X_test = test_csv[config["features"]["numeric"]]
    y_test = test_csv[config["target"]["name"]]
    """

    #Model
    model_classifier_one = RandomForestClassifier(n_estimators=10, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    model_classifier_two = SVC(class_weight='balanced', probability=True, random_state=42)
    model_classifier_three = DecisionTreeClassifier(max_depth=2, random_state=42)


    #Confuse Matrix
    y_keep_habit = (y_train == 1)
    y_train_pred = cross_val_predict(
        model_classifier_one, X_train, y_keep_habit, cv=6
    )
    confusion_matrix_results = confusion_matrix(y_keep_habit, y_train_pred)
    print(f"\nConfusion matrix results: \n{confusion_matrix_results}")

    #Decision Function
    y_scores = cross_val_predict(model_classifier_one, X_train, y_keep_habit, cv=6, n_jobs=-1, method="predict_proba")[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_keep_habit, y_scores)
    plt.plot(thresholds, precisions[:-1], "b--", label="precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g--", label="recall", linewidth=2)
    plt.axvline(x=0, color="k", linewidth=2)
    plt.show()

    idx_90_precision = np.argmax(precisions >= 0.90)
    thresholds_90_precision = thresholds[idx_90_precision]

    y_train_pred_90 = (y_scores >= thresholds_90_precision)
    precision_score_90 = precision_score(y_keep_habit, y_train_pred_90)
    recall_score_90 = recall_score(y_keep_habit, y_train_pred_90)
    print(f"90% precision: {thresholds_90_precision:.4f}")
    print(f"Precision: {precision_score_90:.4f}")
    print(f"Recall: {recall_score_90:.4f}")

if __name__ == "__main__":
    main()
