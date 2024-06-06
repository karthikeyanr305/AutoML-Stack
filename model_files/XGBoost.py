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

from imblearn.over_sampling import SMOTE

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def XGBoost(X, y, isSMOTE):

    X_train, X_test, y_train, y_test = None, None, None, None

    #perform SMOTE on dataset
    
    if isSMOTE:
        smt = SMOTE(random_state=20)
        X_resampled, Y_resampled = smt.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns = list(X.columns))
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.20, random_state=20, stratify= Y_resampled)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, shuffle=True, stratify= y)
    
    

    xgb_classifier = xgb.XGBClassifier()

    xgb_classifier.fit(X_train, y_train)
    joblib.dump(xgb_classifier,"models/xgb_classifier.joblib")

    '''# Setup the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of gradient boosted trees. Equivalent to number of boosting rounds
        'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage used to prevent overfitting. Range is [0,1]
        'max_depth': [3, 4, 6],  # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit
        'colsample_bytree': [0.3, 0.7, 1],  # The fraction of features to use per tree. This is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed
        'subsample': [0.5, 0.7, 1]  # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration
    }
    st.write("Performing Grid search to find the best model!")
    # Setup the grid search
    grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Best parameters found
    print("Best parameters:", grid_search.best_params_)

    # Best score achieved
    print("Best score:", grid_search.best_score_)

    # Optional: Use the best model to make predictions
    best_xgb = grid_search.best_estimator_'''

    return xgb_classifier, X, y, X_train, X_test, y_train, y_test