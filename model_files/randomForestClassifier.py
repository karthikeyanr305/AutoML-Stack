import streamlit as st
import pandas as pd
import numpy as np
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
import joblib
from sklearn.model_selection import GridSearchCV

 #perform SMOTE on dataset
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

def randomForestClassifier(X, y, isSMOTE):

    X_train, X_test, y_train, y_test = None, None, None, None
    
    if isSMOTE:
        smt = SMOTE(random_state=20)
        X_resampled, Y_resampled = smt.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20, stratify= Y_resampled)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True, stratify= y)

    

    rm = RandomForestClassifier(n_estimators=10, max_depth=15, criterion="gini", min_samples_split=10)
    rm.fit(X_train, y_train)
    joblib.dump(rm,"models/RF.joblib")

    '''# Setup the parameter grid
    param_grid = {
        'n_estimators': [10, 20, 50, 80],  # number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # minimum number of samples required to be at a leaf node
    }

    st.write("Optimizing model parameters using grid search!")
    # Setup the grid search
    grid_search = GridSearchCV(rm, param_grid, cv=5)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Best parameters found
    print("Best parameters:", grid_search.best_params_)

    # Best score achieved
    print("Best score:", grid_search.best_score_)

    # Use the best model to make predictions
    best_rf = grid_search.best_estimator_'''


    return rm, X, y, X_train, X_test, y_train, y_test