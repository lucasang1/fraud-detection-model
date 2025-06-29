from joblib import dump
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing import drop_unused, split_X_y

def main():
    # load and remove rows with null values
    df_full = pd.read_csv("cc_statistics.csv")
    df = df_full.dropna(subset=['prev_amount','secs_since_prev','roll_mean_5','roll_sd_5','hour'])
    df = drop_unused(df)

    # split into training and testing sets (80/20)
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )

    # pipeline contains scaler and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state = 42, class_weight = 'balanced'))
    ])

    # train model
    pipeline.fit(X_train, y_train)

    # evaluate (1.0 = perfect. 0.5 = random guessing)
    y_predict_proba = pipeline.predict_proba(X_test)[:,1] # frauds only
    print("ROC-AUC:", roc_auc_score(y_test, y_predict_proba))
    print("PR-AUC:", average_precision_score(y_test, y_predict_proba))

    # save pipeline
    dump(pipeline, "pipeline.pkl")

# ensures that main() isn't run on other files where train.py is imported
if __name__ == "__main__": main()
