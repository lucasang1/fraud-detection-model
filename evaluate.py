from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (confusion_matrix, roc_curve, average_precision_score, precision_recall_curve, classification_report)
from sklearn.model_selection import train_test_split

from preprocessing import drop_unused, split_X_y

def main():
    # load data without rows with null values, model and scaler
    df_full = pd.read_csv("cc_statistics.csv")
    df = df_full.dropna(subset=['prev_amount','secs_since_prev','roll_mean_5','roll_sd_5','hour'])
    print(f"Dropped {len(df_full) - len(df)} rows due to missing values")
    pipeline = load("pipeline.pkl")

    # split into training and testing sets (80/20)
    df = drop_unused(df)
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )

    # evaluate (1.0 = perfect. 0.5 = random guessing)
    y_predict = pipeline.predict(X_test)
    y_predict_proba = pipeline.predict_proba(X_test)[:,1] # frauds only
    
    # classification report + confusion matrix
    print(classification_report(y_test, y_predict))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))

    # plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_predict_proba) # false and true +ve rates
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    # # get PR-AUC value and plot precision-recall curve
    pr_auc = average_precision_score(y_test, y_predict_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_predict_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    # shapley value explains RandomForest and then plot
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'], data=X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    # summary plot
    shap.summary_plot(shap_values[:, :, 1], X_test_scaled, feature_names=X_test.columns, max_display = 29)

if __name__ == "__main__":
    main()