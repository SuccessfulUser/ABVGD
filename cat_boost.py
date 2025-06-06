import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('dataset.csv')

X = data.drop(['Is_Fraud', 'Customer_ID', 'Transaction_ID', 'Customer_Name', 'Customer_Email',
               'Customer_Contact', 'Transaction_Description', 'Merchant_ID'], axis=1)
y = data['Is_Fraud']

X['Transaction_Date'] = pd.to_datetime(X['Transaction_Date'], format='%d-%m-%Y')
X['Transaction_Hour'] = pd.to_datetime(X['Transaction_Time'], format='%H:%M:%S').dt.hour
X['Transaction_DayOfWeek'] = X['Transaction_Date'].dt.dayofweek
X['Transaction_Month'] = X['Transaction_Date'].dt.month
X = X.drop(['Transaction_Date', 'Transaction_Time', 'Transaction_Location'], axis=1)

categorical_features = ['Gender', 'State', 'City', 'Bank_Branch', 'Account_Type',
                       'Transaction_Type', 'Merchant_Category', 'Transaction_Currency',
                       'Transaction_Device', 'Device_Type', 'Transaction_Hour',
                       'Transaction_DayOfWeek', 'Transaction_Month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    scale_pos_weight=37982/2018,
    cat_features=categorical_features,
    verbose=100,
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({
    'Признак': X_train.columns,
    'Важность': model.get_feature_importance()
})
print("\nВажность признаков (Топ 10):\n", feature_importance.sort_values(by='Важность', ascending=False).head(10))

model.save_model('fraud_detection_catboost_model.cbm')
