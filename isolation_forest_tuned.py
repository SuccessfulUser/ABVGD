import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
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
fraud_rate = data.groupby('Merchant_Category')['Is_Fraud'].mean().reset_index(name='Merchant_Fraud_Rate')
data = data.merge(daily_counts, on=['Customer_ID', 'Transaction_Date_Str'], how='left')
data = data.merge(daily_avg_amount, on=['Customer_ID', 'Transaction_Date_Str'], how='left')
data = data.merge(fraud_rate, on='Merchant_Category', how='left')

X = data.drop(['Is_Fraud', 'Customer_ID', 'Transaction_ID', 'Customer_Name', 'Customer_Email',
               'Customer_Contact', 'Transaction_Description', 'Merchant_ID', 'Transaction_Date',
               'Transaction_Time', 'Transaction_Location', 'Transaction_Date_Str'], axis=1)
y = data['Is_Fraud']

categorical_features = ['Gender', 'State', 'City', 'Bank_Branch', 'Account_Type', 'Transaction_Type',
                       'Merchant_Category', 'Transaction_Currency', 'Transaction_Device', 'Device_Type',
                       'Transaction_Hour', 'Transaction_DayOfWeek', 'Transaction_Month', 'Is_Night_Transaction']
numeric_features = ['Age', 'Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count', 'Daily_Avg_Amount', 'Merchant_Fraud_Rate']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train_preprocessed)

anomaly_scores = -model.score_samples(X_test_preprocessed)

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
optimal_threshold = thresholds[np.argmax(2 * precision * recall / (precision + recall + 1e-10))]
y_pred = (anomaly_scores > optimal_threshold).astype(int)

print("F1-Score:", f1_score(y_test, y_pred))
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, 'fraud_detection_isolation_forest_model.pkl')
joblib.dump(preprocessor, 'isolation_forest_preprocessor.pkl')
