import os
import mysql.connector 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error, accuracy_score, auc, classification_report, confusion_matrix, roc_curve, RocCurveDisplay

df=pd.read_csv('ASD.csv')

#print(df.head())
#print(df.tail())
'''print(df.info())
print(df.shape)

print(df.isnull().sum())
print(df.nunique())'''
print(df.columns.to_list())




# DAta Cleaning

# Columns to convert
'''bool_columns = [
    'IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative',
    'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist',
    'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism'
]

# Convert TRUE/FALSE to 1/0
df[bool_columns] = df[bool_columns].replace({True: 1, False: 0})


print("Conversion completed successfully!")
'''




#EDA Analysis

hate_columns = [
    'IsToxic','IsAbusive','IsThreat','IsProvocative',
    'IsObscene','IsHatespeech','IsRacist','IsNationalist',
    'IsSexist','IsHomophobic','IsReligiousHate','IsRadicalism'
]

hate_counts = df[hate_columns].sum()

'''hate_counts.plot(kind='bar', figsize=(10,5))
plt.title('Count of Hate Categories')
plt.ylabel('Number of Comments')
plt.xticks(rotation=45)
plt.show()
'''

#Bivariate Analysis
#plt.figure(figsize=(8, 5))
'''sns.violinplot(data=df, x="IsToxic", y="IsAbusive",  palette="Set2")

plt.title("IsToxic vs. IsAbusive")
plt.xlabel("IsToxic")
plt.ylabel("IsAbusive")
#plt.tight_layout()
plt.show()
'''

#Multivariate Analysis

'''plt.figure(figsize=(12,8))
sns.heatmap(df[hate_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Hate Categories')
plt.show()'''

#Create total hate score
df['Total_Hate_Score'] = df[hate_columns].sum(axis=1)
'''sns.histplot(df['Total_Hate_Score'], bins=10, kde=True)
plt.title('Distribution of Total Hate Score')
plt.show()'''

'''
#Percentage of hate vs non-hate comments
hate_present = (df[hate_columns].sum(axis=1) > 0).value_counts()
hate_present.plot(kind='pie', autopct='%1.1f%%')
plt.title('Hate vs Non-Hate Comments')
plt.show()
'''

#Linear Regression 
features = [
    'IsAbusive','IsThreat','IsProvocative','IsObscene',
    'IsRacist','IsNationalist','IsSexist',
    'IsHomophobic','IsReligiousHate','IsRadicalism'
]

'''X = df[features]
y = df['Total_Hate_Score']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2_score=r2_score(y_test,y_pred)

print('Mean Squared Error:',mse)
print('R2 Score:',r2_score)'''

#Logistic Regression

df['Hate_Label'] = (df['Total_Hate_Score'] > 0).astype(int)

'''y = df['Hate_Label']
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Example 1")
plt.legend()
plt.show()

'''





def to_bool(value):
    if isinstance(value, str):
        return 1 if value.strip().lower() == "true" else 0
    return int(value)



#MySql Connection and Database Creation

'''db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Anjali@sql'
)
mycursor = db.cursor()
mycursor.execute("CREATE DATABASE IF NOT EXISTS Automated_hate_speech")
db.close()'''



# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Anjali@sql",
    database="Automated_hate_speech"
)

if db.is_connected():
    print("✅ Connected to MySQL successfully")

db.close()






import csv
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Anjali@sql",
    database="Automated_hate_speech"
)

cursor = db.cursor()

sql = '''
INSERT INTO HSD
(CommentId, VideoId, Text,
 IsToxic, IsAbusive, IsThreat, IsProvocative, IsObscene,
 IsHatespeech, IsRacist, IsNationalist, IsSexist,
 IsHomophobic, IsReligiousHate, IsRadicalism)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
'''

with open("ASD.csv", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        # Convert all boolean columns (index 3 to 14)
        for i in range(3, 15):
            row[i] = to_bool(row[i])

        cursor.execute(sql, row)

db.commit()
cursor.close()
db.close()

print("CSV data inserted into MySQL successfully!")


