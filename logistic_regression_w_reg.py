import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

columns_to_drop = [
    'Customer_ID', 'Customer_Name', 'Transaction_ID',
    'Transaction_Date', 'Transaction_Time', 'Merchant_ID',
    'Transaction_Description', 'Customer_Email'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

X = df.drop('Is_Fraud', axis=1)
y = df['Is_Fraud']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred))

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.show()

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=['Не мошенничество', 'Мошенничество'])
