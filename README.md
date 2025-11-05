# Data-Science-Projects

1. ETL Development Project
## Dataset: Online retail & E-commerce Dataset
## Source: https://www.kaggle.com/datasets/ertugrulesol/online-retail-data

Code:
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset / Read csv file
df = pd.read_csv(r'C:\Users\user\Desktop\Study Notes\MDS502\synthetic_online_retail_data.csv')

# View first 5 rows by default
df.head()

# Checking for missing values
missing_values = df.isnull().sum()
missing_values[missing_values > 0]

# Boxplot to detect outliers
for col in ['revenue']:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

# Define IQR-based outlier detection
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

df_no_outliers = remove_outliers(df, 'revenue')

# Split date and time
df['order_datetime'] = pd.to_datetime(df['order_date'])
df['order_date'] = df['order_datetime'].dt.date
df['order_time'] = df['order_datetime'].dt.time

# View first 5 rows by default
df.head()

# Standardization
scaler = MinMaxScaler()
df[['quantity', 'price', 'revenue']] = scaler.fit_transform(df[['quantity', 'price', 'revenue']])

# View first 5 rows by default
df.head()

# Results/Model Evaluation Through Factor Analysis/classification
# Selected relevant variables for PCA
features = ['quantity', 'price', 'customer_id', 'revenue']
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create DataFrame with principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Visualize PCA results
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', alpha=0.5)
plt.title('PCA of Retail Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Evaluation via  classfication (High vs Low Revenue)
df['revenueclass'] = pd.qcut(df['revenue'], q=2, labels=['Low', 'High'])

X = df[['quantity', 'price']]
y = df['revenueclass']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print results
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Retained variables based on influence and redundancy
df_final = df[['quantity', 'price', 'revenue', 'city', 'product_name', 'order_date']]
df_final.head()

# Load data into MySQL via Python
pip install mysql-connector-python
import csv
import mysql.connector

#Establish connection
conn = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    password='root'
)

# Create a cursor object
cursor = conn.cursor()

csv_data = csv.reader(r"C:\Users\user\Desktop\Study Notes\MDS502\synthetic_online_retail_data_og.csv")

for row in csv_data:
    cursor.execute('INSERT INTO ECOMMERCE(customer_id, order_date, product_id, \
        category_id, category_name, product_name, quantity, price, payment_method, \
        city, review_score, gender, age, revenue,'\
        'VALUES("%d", "%s", "%d", "%d", "%s", "%s", "%d", "%f", "%s", "%s", \
        "%d", "%s", "%d", "%d")',
        row)

# Close the connection
mydb.commit()
cursor.close()
print("Done")
```

