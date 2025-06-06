import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, precision_recall_curve, auc
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from imblearn.over_sampling import SMOTE
import joblib
import xgboost
import warnings
warnings.filterwarnings('ignore')


if xgboost.__version__ < '2.0.0':
    raise ValueError("This code requires XGBoost 2.0.0 or later. Please update with 'pip install --upgrade xgboost'.")

data = pd.read_csv('dataset.csv')

data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'], format='%d-%m-%Y')
data['Transaction_Hour'] = pd.to_datetime(data['Transaction_Time'], format='%H:%M:%S').dt.hour
data['Transaction_DayOfWeek'] = data['Transaction_Date'].dt.dayofweek
data['Transaction_Month'] = data['Transaction_Date'].dt.month
data['Is_Night_Transaction'] = data['Transaction_Hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
data['Transaction_Date_Str'] = data['Transaction_Date'].dt.strftime('%Y-%m-%d')
daily_counts = data.groupby(['Customer_ID', 'Transaction_Date_Str'])['Transaction_ID'].count().reset_index(name='Daily_Transaction_Count')
daily_avg_amount = data.groupby(['Customer_ID', 'Transaction_Date_Str'])['Transaction_Amount'].mean().reset_index(name='Daily_Avg_Amount')
data = data.merge(daily_counts, on=['Customer_ID', 'Transaction_Date_Str'], how='left')
data = data.merge(daily_avg_amount, on=['Customer_ID', 'Transaction_Date_Str'], how='left')

X = data.drop(['Is_Fraud', 'Customer_ID', 'Transaction_ID', 'Customer_Name', 'Customer_Email',
               'Customer_Contact', 'Transaction_Description', 'Merchant_ID', 'Transaction_Date',
               'Transaction_Time', 'Transaction_Location', 'Transaction_Date_Str'], axis=1)
y = data['Is_Fraud']

categorical_features = ['Gender', 'State', 'Account_Type', 'Transaction_Type', 'Merchant_Category',
                       'Transaction_Currency', 'Transaction_Device', 'Device_Type', 'Transaction_Hour',
                       'Transaction_DayOfWeek', 'Transaction_Month', 'Is_Night_Transaction']
numeric_features = ['Age', 'Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count', 'Daily_Avg_Amount']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(scale_pos_weight=37982/2018, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)

subset_size = min(50000, X_train.shape[0])
subset_indices = np.random.choice(X_train.shape[0], subset_size, replace=False)
X_train_subset = X_train.iloc[subset_indices]
y_train_subset = y_train.iloc[subset_indices]

param_grid = {
    'classifier__max_depth': [3, 6],
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.01, 0.1]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=1)
grid_search.fit(X_train_subset, y_train_subset)
print("Лучшие параметры:", grid_search.best_params_)

best_model = grid_search.best_estimator_
try:
    early_stopping = EarlyStopping(rounds=10, metric_name='logloss')
    best_model.named_steps['classifier'].fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_test_preprocessed, y_test)],
        callbacks=[early_stopping],
        verbose=False
    )
except TypeError as e:
    print(f"Ошибка с callbacks: {e}")
    print("Переход к обучению без ранней остановки.")
    best_model.named_steps['classifier'].fit(X_train_res, y_train_res)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
print("\nPrecision-Recall AUC:", auc(recall, precision))

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
optimal_threshold = thresholds[np.argmax(2 * precision * recall / (precision + recall + 1e-10))]
y_pred_adjusted = (y_pred_proba > optimal_threshold).astype(int)
print("\nС оптимизированным порогом (оптимальный F1-скор):")
print("F1-Score:", f1_score(y_test, y_pred_adjusted))
print("\nОтчет о классификации (Оптимизированный порог):\n", classification_report(y_test, y_pred_adjusted))
print("\nМатрица ошибок (Оптимизированный порог):\n", confusion_matrix(y_test, y_pred_adjusted))

feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = (numeric_features +
                 best_model.named_steps['preprocessor'].named_transformers_['cat']
                 .get_feature_names_out(categorical_features).tolist())
importance_df = pd.DataFrame({'Признак': feature_names, 'Важность': feature_importance})
print("\nВажность признаков (Топ 10):\n", importance_df.sort_values(by='Важность', ascending=False).head(10))

joblib.dump(best_model, 'fraud_detection_gradient_boosting_smote_optimized_model.pkl')
