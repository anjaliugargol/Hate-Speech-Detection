# 🚀 Automated Hate Speech Detection on Twitter & YouTube Comments

---

## 📌 Project Objective

Analyze social media comments from Twitter and YouTube to automatically detect hate or toxic content and support real-time content moderation using machine learning.

---

## 🧾 Problem Statement

Social media platforms often face challenges in monitoring large volumes of user-generated content, leading to the spread of toxic, abusive, or hateful comments.  
This project aims to analyze comment data, identify different hate categories, and build a predictive model to classify comments as hate or non-hate for proactive moderation.

---

## 📊 Dataset Description

The dataset includes labeled social media comments with multiple hate indicators:

- CommentId  
- VideoId  
- Text  
- IsToxic  
- IsAbusive  
- IsThreat  
- IsProvocative  
- IsObscene  
- IsHatespeech  
- IsRacist  
- IsNationalist  
- IsSexist  
- IsHomophobic  
- IsReligiousHate  
- IsRadicalism  

Additional Features Created:
- Total_Hate_Score  
- Hate_Label (0 = Non-Hate, 1 = Hate)  

---

## 🛠 Tools & Technologies Used

- Python (Pandas, NumPy)
- Matplotlib & Seaborn
- Scikit-learn
- MySQL
- Power BI
- Jupyter Notebook

---

## 🔎 Step 1: Data Cleaning

- Loaded CSV dataset
- Verified null values
- Converted boolean hate indicators into numeric format (0/1)
- Prepared dataset for modeling

---

## 📈 Step 2: Exploratory Data Analysis

- Distribution of hate categories
- Correlation heatmap
- Total Hate Score distribution
- Toxic vs Abusive relationship analysis

---

## 🤖 Step 3: Predictive Modeling

### Logistic Regression

Target Variable:  
Hate_Label (0 = Non-Hate, 1 = Hate)

Evaluation Metrics:
- Accuracy
- Confusion Matrix
- Classification Report
- ROC Curve

---

## 🗄 Step 4: Database Integration

- Created MySQL database schema
- Inserted cleaned dataset into MySQL
- Enabled SQL-based hate analysis

---

## 📊 Step 5: Power BI Dashboard

- Hate category distribution
- Hate vs Non-Hate comparison
- Correlation visualization
- Severity analysis using Total Hate Score

Dashboard File:
`/dashboard/Hate_speech.pbix`

---

## 📌 Key Insights

- Automated hate speech detection helps social media platforms proactively filter toxic comments, improving user safety and engagement.
- High co-occurrence of abusive, racist, and religious hate content indicates the need for multi-category moderation strategies.
- The Total Hate Score enables prioritization of highly offensive comments for faster review and action.
- Binary classification of comments into hate vs non-hate supports real-time automated content moderation systems.
- Data-driven hate speech analysis assists organizations in strengthening community guidelines and compliance monitoring.

---

## 🚀 Future Enhancements

- Real-time API integration
- Deep learning models (LSTM / BERT)
- Live moderation alert system

---

## 📂 Project Structure

Automated-Hate-Speech-Detection/
│
├── data/
├── notebooks/
├── database/
├── dashboard/
├── requirements.txt
└── README.md

---

## ▶ How to Run the Project

1. Clone the repository  
2. Install dependencies
3. Open `notebooks/hate_speech_analysis.ipynb`
4. Run all cells
5. Execute `database/create_database.sql`
6. Open dashboard in Power BI
