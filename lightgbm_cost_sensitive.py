import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, precision_recall_curve
from lightgbm import LGBMClassifier, early_stopping
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

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

categorical_features = ['Gender', 'State', 'City', 'Bank_Branch', 'Account_Type', 'Transaction_Type',
                       'Merchant_Category', 'Transaction_Currency', 'Transaction_Device', 'Device_Type',
                       'Transaction_Hour', 'Transaction_DayOfWeek', 'Transaction_Month', 'Is_Night_Transaction']
numeric_features = ['Age', 'Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count', 'Daily_Avg_Amount']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
    ('undersample', RandomUnderSampler(sampling_strategy=0.8, random_state=42)),
    ('classifier', LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=37982/2018,
        random_state=42,
        verbose=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
optimal_threshold = thresholds[np.argmax(2 * precision * recall / (precision + recall + 1e-10))]
y_pred_adjusted = (y_pred_proba > optimal_threshold).astype(int)
print("\nС оптимизированным порогом (оптимальный F1-скор):")
print("F1-Score:", f1_score(y_test, y_pred_adjusted))
print("\nОтчет о классификации (Оптимизированный порог):\n", classification_report(y_test, y_pred_adjusted))
print("\nМатрица ошибок (Оптимизированный порог):\n", confusion_matrix(y_test, y_pred_adjusted))

model = pipeline.named_steps['classifier']
feature_names = (numeric_features +
                 pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
})
print("\nВажность признаков (Топ 10):\n", feature_importance.sort_values(by='Importance', ascending=False).head(10))

joblib.dump(pipeline, 'fraud_detection_lightgbm_focal_model.pkl')