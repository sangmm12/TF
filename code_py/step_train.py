import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from joblib import dump
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 只使用第一个 GPU

# Set display option to show all columns
pd.set_option('display.max_columns', 15)

data_model = 'model_5'
version = 'H47'#'M36'#'H47'
hm = 'human'#'mouse'#'human'
dim = 2736#1368#2736
X_start = 6#4#5 #3/4 for model 1/2 5/6 for model 3/4/5/6 For Type increase by 1

df = pd.read_csv(f'TF_{hm}_{version}_{data_model}_NV_{dim}_Type.csv')
#df = pd.read_csv(f'TF_human_H47_model_1_NV_1368_Type.csv')
#df = pd.read_csv(f'TF_human_H47_model_2_NV_1368_Type.csv')
print(df.head())
print(df.shape)
print(df.columns)
'''
# Handle data imbalance by taking equal number of samples from both classes
#print("Handling data imbalance...")
class_0 = df[df['Interaction'] == True]
class_1 = df[df['Interaction'] == False]

min_class_size = min(len(class_0), len(class_1))
class_0_sample = class_0.sample(min_class_size, random_state=42)
class_1_sample = class_1.sample(min_class_size, random_state=42)

balanced_df = pd.concat([class_0_sample, class_1_sample]).sample(frac=1, random_state=42)
print("Balanced dataset shape:", balanced_df.shape)

# Separate features and target after balancing
X_balanced = balanced_df.iloc[:, X_start:]
y_balanced = balanced_df['Interaction']
print(X_balanced.shape)
'''
# Using predefined train, validation, and test sets
X_train = df.iloc[:, X_start:]
y_train = df['Interaction']
#X_train = df[df['Type'] == 'train'].iloc[:, X_start:]
#y_train = df[df['Type'] == 'train']['Interaction']
X_val = df[df['Type'] == 'val'].iloc[:, X_start:]
y_val = df[df['Type'] == 'val']['Interaction']
X_test = df[df['Type'] == 'test'].iloc[:, X_start:]
y_test = df[df['Type'] == 'test']['Interaction']

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

'''
# Random Forest Classifier Hyperparameter Tuning
print("==========Starting hyperparameter tuning for Random Forest Classifier==========")
rf_model = RandomForestClassifier(random_state=42)

rf_param_grid = {
    'n_estimators': [500, 1000, 1200, 1400, 1600, 1800, 2000],
    'max_depth': [20, 50, 100],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [2, 4, 6, 8]
}

rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)
print("Random Forest hyperparameter tuning completed.")
best_rf_model = rf_grid_search.best_estimator_
print("Best Random Forest parameters found:", rf_grid_search.best_params_)
dump(best_rf_model, f'best_models/best_rf_{hm}_{version}_{data_model}.pkl')

# Output Random Forest Grid Search Results
results = rf_grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"ROC AUC Score: {mean_score:.4f}, Parameters: {params}")

# Gradient Boosting Classifier Hyperparameter Tuning
print("==========Starting hyperparameter tuning for Gradient Boosting Classifier==========")
gb_model = GradientBoostingClassifier(random_state=42)
gb_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.1, 0.15],
    'max_depth': [5, 10]
}

gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
gb_grid_search.fit(X_train, y_train)
print("Gradient Boosting hyperparameter tuning completed.")
best_gb_model = gb_grid_search.best_estimator_
print("Best Gradient Boosting parameters found:", gb_grid_search.best_params_)
dump(best_gb_model, f'best_models/best_gb_{hm}_{version}_{data_model}.pkl')

# Output Gradient Boosting Grid Search Results
results = gb_grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"ROC AUC Score: {mean_score:.4f}, Parameters: {params}")
'''
# Hyperparameter Tuning for XGBoost
print("==========Starting hyperparameter tuning for XGBoost==========")

# XGBoost Classifier with GPU support
print("Training XGBoost Classifier with GPU support")
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, tree_method='hist', device='cuda')
try:
    xgb_model.fit(X_train, y_train)
    print("XGBoost training completed.")
except xgb.core.XGBoostError as e:
    print("XGBoost training failed. Error:", e)
    print("Falling back to CPU training...")
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, tree_method='hist')
    xgb_model.fit(X_train, y_train)
    print("XGBoost CPU training completed.")

xgb_param_grid = {
    'n_estimators': [250, 350, 500, 550],
    'max_depth': [5, 20, 40, 60],
    'learning_rate': [0.05, 0.1, 0.15]
}
'''
xgb_param_grid = {
    'n_estimators': [550],
    'max_depth': [40],
    'learning_rate': [0.15]
}
'''
try:
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)
    print("Hyperparameter tuning completed.")
    best_xgb_model = grid_search.best_estimator_
    print("Best XGBoost parameters found:", grid_search.best_params_)
except Exception as e:
    print("Hyperparameter tuning failed. Error:", e)
    best_xgb_model = xgb_model

# Save the best XGBoost model
#with open(f'best_models/best_xgb_{hm}_{version}_{data_model}.pkl', 'wb') as f:
with open(f'best_xgb_{hm}_{version}_{data_model}.pkl', 'wb') as f:    
    pickle.dump(best_xgb_model, f)

# Model Evaluation
def evaluate_model(model, X_val, y_val, X_test, y_test):
    print("\nEvaluating model...")
    print("Validation Set Evaluation:")
    y_val_pred = model.predict(X_val)
    print(classification_report(y_val, y_val_pred))
    print("Validation AUC-ROC:", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

    print("\nTest Set Evaluation:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    print("Test AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Evaluate Random Forest
#evaluate_model(best_rf_model, X_val, y_val, X_test, y_test)

# Evaluate Gradient Boosting
#evaluate_model(best_gb_model, X_val, y_val, X_test, y_test)

# Evaluate Best XGBoost Model
evaluate_model(best_xgb_model, X_val, y_val, X_test, y_test)

