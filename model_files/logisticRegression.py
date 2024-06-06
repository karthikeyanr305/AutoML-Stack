import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
#from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import seaborn as sb
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, confusion_matrix, auc
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib
import shap

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, f1_score

'''
def plot_shap(model, X_train, X_test):
    # Initialize the SHAP Explainer
    explainer = shap.Explainer(model, X_train)

    # Compute SHAP values for the test set
    shap_values = explainer(X_test)

    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

    
    # Calculate mean absolute SHAP values
    mean_shap = shap_df.abs().mean().sort_values(ascending=False)
    
    # Convert mean_shap Series to DataFrame for better handling
    mean_shap_df = mean_shap.reset_index()
    mean_shap_df.columns = ['Feature', 'Mean SHAP Value']

    # Sort the DataFrame by 'Mean SHAP Value' in descending order
    mean_shap_df = mean_shap_df.sort_values(by='Mean SHAP Value', ascending=False)

    # Create the bar plot using Plotly Express
    fig_shap = px.bar(mean_shap_df, x='Mean SHAP Value', y='Feature',
                labels={'Feature': 'Feature', 'Mean SHAP Value': 'Mean SHAP Value'},
                title='Mean Absolute SHAP Values',
                orientation='h', text='Mean SHAP Value')  # Add text parameter for displaying values

    # Adjust text annotation position
    fig_shap.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # Update layout for a clearer view
    fig_shap.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                    xaxis_title='Mean SHAP Value',
                    yaxis_title='Feature')
    
    st.plotly_chart(fig_shap, use_container_width=True)


def plot_hist_score(y, y_score):
    # The histogram of scores compared to true labels
    fig_hist = px.histogram(
        x=y_score, color=y, nbins=50,
        title='Histogram of Scores',
        labels=dict(color='True Labels', x= 'Score', y= 'Count')
    )

    #fig_hist.show()
    st.plotly_chart(fig_hist, use_container_width=True)

def plot_threshold(fpr, tpr, thresholds):

    # Evaluating model performance at various thresholds
    df_thres = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr
    }, index=thresholds)
    df_thres.index.name = "Thresholds"
    df_thres.columns.name = "Rate"

    fig_thresh = px.line(
        df_thres, title='TPR and FPR at every threshold',
        width=700, height=500
    )

    fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
    #fig_thresh.show()
    st.plotly_chart(fig_thresh, use_container_width=True)


def plot_auc_roc(fpr, tpr):
    fig_auc = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
    )
    fig_auc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig_auc.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_auc.update_xaxes(constrain='domain')
    #fig.show()
    st.plotly_chart(fig_auc, use_container_width=True)

def plot_pr_curve(precision, recall, fpr, tpr):

    fig_pr = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(recall, precision):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=700, height=500
    )
    fig_pr.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig_pr.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_pr.update_xaxes(constrain='domain')

    #fig_pr.show()
    st.plotly_chart(fig_pr, use_container_width=True)

def plot_confusion_matrix(cm):

    # Convert the confusion matrix to DataFrame for better labeling in Plotly Express
    df_cm = pd.DataFrame(cm, index=['Authentic', 'Fraud'], columns=['Predicted Authentic', 'Predicted Fraud'])

    # Create the heatmap using Plotly Express
    fig_cm = px.imshow(df_cm,
                    labels=dict(x="Predicted Label", y="True Label", color="Number of Samples"),
                    x=['Authentic', 'Fraud'],
                    y=['Authentic', 'Fraud'],
                    text_auto=True)  # This will annotate the heatmap with the numeric values

    # Add titles
    fig_cm.update_layout(title="Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label")

    # Show the plot
    #fig_cm.show()
    st.plotly_chart(fig_cm, use_container_width=True)'''

def logisticRegression(X, y, isSMOTE):
    #st.write("Entered logisticRegression")
    
    
    X_train, X_test, y_train, y_test = None, None, None, None
     
    #perform SMOTE on dataset if chosen
    if isSMOTE:
        smt = SMOTE(random_state=20)
        #st.write("00")
        X_resampled, Y_resampled = smt.fit_resample(X, y)
        #st.write("000")
        X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
        #X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20, stratify= Y_resampled)  
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20, stratify= Y_resampled)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True, stratify= y)
    
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    joblib.dump(LR,"models/LR.joblib")

    
    return LR, X, y, X_train, X_test, y_train, y_test



