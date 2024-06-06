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
from joblib import dump, load
import shap

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, f1_score
import json

@st.cache_data
def plot_shap(_model, X_train, X_test, model_name):
    # Initialize the SHAP Explainer
    
    explainer = None
    
    if model_name == 'LR':
        explainer = shap.Explainer(_model, X_train)
    else:
        explainer = shap.TreeExplainer(_model)

    # Compute SHAP values for the test set
    shap_values = explainer(X_test)

    # Convert SHAP values to DataFrame
    shap_df = None
    if model_name in ['LR', 'XG']:
        shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
    elif model_name in ['RF']:
        shap_df = pd.DataFrame(shap_values.values[:,:,1], columns=X_test.columns)

    
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


@st.cache_data
def plot_hist_score(y, y_score):
    # The histogram of scores compared to true labels
    fig_hist = px.histogram(
        x=y_score, color=y, nbins=50,
        title='Histogram of Scores',
        labels=dict(color='True Labels', x= 'Score', y= 'Count')
    )

    #fig_hist.show()
    st.plotly_chart(fig_hist, use_container_width=True)

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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
    st.plotly_chart(fig_cm, use_container_width=True)


def visualize_model(model, X, y, X_train, X_test, y_train, y_test, model_name, isSMOTE, file_size):

    st.write("Optimizing model parameters using grid search!")

    model_pred = model.predict(X_test)
    classification_report_str = classification_report(y_test, model_pred)
    accuracy_model = accuracy_score(y_test, model_pred)
    precision_model = precision_score(y_test, model_pred)
    recall_model = recall_score(y_test, model_pred)
    # F1 Score: The weighted average of Precision and Recall.
    f1_model = f1_score(y_test, model_pred)

    st.markdown("<h3 style='text-align: center;color: #5fb4fb;'><u>METRICS</u></h3>", unsafe_allow_html=True)
    #st.text(classification_report_str)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Accuracy", f"{accuracy_model:.2f}")
    metric2.metric("Precision", f"{precision_model:.2f}")
    metric3.metric(" Recall", f"{recall_model:.2f}")
    metric4.metric("F-1 Score", f"{f1_model:.2f}")

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    #st.markdown('<h3 style="color: #5fb4fb;">PLOTS:</h3>', unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;color: #5fb4fb;'><u>PLOTS</u></h3>", unsafe_allow_html=True)

    #Histogram of Scores
    y_score = model.predict_proba(X)[:, 1]

    # AUC ROC 
    fpr, tpr, thresholds = roc_curve(y, y_score)

    # Precision Recall Curve
    precision, recall, thresholds2 = precision_recall_curve(y, y_score)

    # Confusion Matrix
    cm = confusion_matrix(y_test, model_pred)

    '''st.write(cm)
    st.write('y_test.size():' , y_test.size)
    st.write('sum(y_test)', sum(y_test))

    st.write('y_train.size():' , y_train.size)
    st.write('sum(y_train)', sum(y_train))'''

    top_left, top_right = st.columns(2)
    mid_left, mid_right = st.columns(2)
    bottom_left, bottom_right  = st.columns(2)



    with top_left:
        plot_shap(model, X_train, X_test, model_name)
        
        #st.write("block for shap")

    with top_right:
        plot_hist_score(y, y_score)

    with mid_left:
        plot_threshold(fpr, tpr, thresholds)
    
    with mid_right:
        plot_confusion_matrix(cm)
    
    with bottom_left:
        plot_auc_roc(fpr, tpr)

    with bottom_right:
        plot_pr_curve(precision, recall, fpr, tpr)

    

    model_metrics ={}
    model_metrics['metric_names'] = ('Accuracy', 'Precision', 'Recall', 'F-1 Score')
    model_metrics['metric_values'] = (accuracy_model, precision_model, recall_model, f1_model)

    # Writing JSON data
    with open('models/{}_metrics_{}_{}.json'.format(model_name, file_size, isSMOTE), 'w') as f:
        json.dump(model_metrics, f, indent=4) 

    
    return (accuracy_model, precision_model, recall_model, f1_model)
