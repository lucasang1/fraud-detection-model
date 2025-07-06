from joblib import load
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import streamlit as st

from preprocessing import drop_unused, split_X_y

# set up page layout
st.set_page_config(layout = "centered")
st.title("Fraud Detection Model Dashboard")

# load data
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv("cc_statistics.csv")
    return df
df_full = load_data("cc_statistics.csv")
df = df_full.dropna(subset = ['prev_amount', 'secs_since_prev', 'roll_mean_5', 'roll_sd_5', 'hour'])
df = drop_unused(df)
X, y = split_X_y(df)

# load model
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return load(path)
pipeline = load_model("pipeline.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_predict = pipeline.predict(X_test)
y_predict_proba = pipeline.predict_proba(X_test)[:, 1]

# sidebar navigation
view = st.sidebar.radio("Choose view:", ["Metrics", "ROC and PR Curves", "SHAP Analysis", "Predict Single", "Predict Batch", "Feature Importance"])

if view == "Metrics":
    st.header("Performance Metrics")
    st.subheader("Transactions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(y_test))
    col2.metric("Total Fradulent Transactions", y_test.sum(), f"{y_test.sum()/len(y_test):.2%}")
    col3.metric("Flagged as Fraud", y_predict.sum(), f"{y_predict.sum()/len(y_test):.2%}")

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_predict, target_names = ["Non-Fraud (0)", "Fraud (1)"], output_dict = True)).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_predict)
    cm_norm = cm.astype(float) / cm.sum(axis = 1)[:, np.newaxis]
    
    tn, fp, fn, tp = cm.ravel()
    col4, col5, col6 = st.columns(3)
    col4.metric("Total Inaccurate Predictions", fp + fn, f"{(fp + fn)/len(y_test):.2%}")
    col5.metric("Falsely Flagged as Fraud (FP)", fp)
    col6.metric("Fraud Not Flagged (FN)", fn)

    fig, ax = plt.subplots()
    sns.heatmap(cm_norm, annot = cm, fmt = "d", cmap = 'PuBu', cbar = False, xticklabels = ["False", "True"], yticklabels = ["False", "True"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

elif view == "ROC and PR Curves":
    st.header("ROC and Precision-Recall Curves")
    roc_auc = roc_auc_score(y_test, y_predict_proba)
    fpr, tpr, _ = roc_curve(y_test, y_predict_proba)
    fig_roc = px.area(x = fpr, y = tpr, labels = {'x': 'False Positive Rate', 'y': 'True Positive Rate'}, title = f'ROC Curve (AUC = {roc_auc:.3f})')
    st.plotly_chart(fig_roc)

    precision, recall, _ = precision_recall_curve(y_test, y_predict_proba)
    pr_auc = average_precision_score(y_test, y_predict_proba)
    fig_pr = px.area(x = recall, y = precision, labels = {'x' : 'Recall', 'y': 'Precision'}, title = f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
    st.plotly_chart(fig_pr)

elif view == "SHAP Analysis":
    st.header("Model Explanability with SHAP")
    st.subheader("SHAP Summary Plot")
    st.image("shap_summary_plot.png", use_container_width = True)

elif view == "Predict Single":
    st.header("Predict Single Record")
    input_data = {}
    for col in X.columns: 
        val = st.number_input(f"{col}", value = 0.0)
        input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df_scaled = pipeline.named_steps['scaler'].transform(input_df)
        predict_proba = pipeline.named_steps['classifier'].predict_proba(input_df_scaled)[0, 1]
        st.metric("Fraud Probability", f"{predict_proba:.2%}")

elif view == "Predict Batch":
    st.header("Predict Batch CSV")
    file = st.file_uploader("Upload CSV", type = ['csv'])
    if file:
        df_input = pd.read_csv(file)
        df_input = drop_unused(df_input.dropna())
        preds = pipeline.predict_proba(df_input)[:, 1]
        df_input['Fraud Probability'] = preds
        st.dataframe(df_input)
        st.download_button('Download Predictions', df_input.to_csv(index = False), "predictions.csv")

elif view == "Feature Importance":
    st.header("Feature Importance")
    importances = pipeline.named_steps['classifier'].feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns,'Importance': importances}).sort_values(by = 'Importance', ascending = True)
    fig_feat = px.bar(feat_df, x = "Importance", y = 'Feature', orientation = 'h')
    st.plotly_chart(fig_feat)

st.sidebar.markdown("---")
st.sidebar.write(f"{len(df_full) - len(df)} rows dropped due to missing values.")