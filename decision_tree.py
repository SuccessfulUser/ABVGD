import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
import warnings
import joblib
warnings.filterwarnings('ignore')

data = pd.read_csv('dataset.csv')

X = data.drop(['Is_Fraud', 'Customer_ID', 'Transaction_ID', 'Customer_Name', 'Customer_Email',
               'Customer_Contact', 'Transaction_Description', 'Merchant_ID'], axis=1)
y = data['Is_Fraud']

X['Transaction_Date'] = pd.to_datetime(X['Transaction_Date'], format='%d-%m-%Y')
X['Transaction_Hour'] = pd.to_datetime(X['Transaction_Time'], format='%H:%M:%S').dt.hour
X['Transaction_DayOfWeek'] = X['Transaction_Date'].dt.dayofweek
X['Transaction_Month'] = X['Transaction_Date'].dt.month
X = X.drop(['Transaction_Date', 'Transaction_Time'], axis=1)

X = X.drop('Transaction_Location', axis=1)

categorical_features = ['Gender', 'State', 'City', 'Bank_Branch', 'Account_Type',
                       'Transaction_Type', 'Merchant_Category', 'Transaction_Currency',
                       'Transaction_Device', 'Device_Type', 'Transaction_Hour',
                       'Transaction_DayOfWeek', 'Transaction_Month']
numeric_features = ['Age', 'Transaction_Amount', 'Account_Balance']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42,
                                         max_depth=10, min_samples_split=10))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))

joblib.dump(pipeline, 'fraud_detection_decision_tree_model.pkl')
