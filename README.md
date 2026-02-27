```markdown
# Automated Hate Speech Detection on Twitter & YouTube Comments

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
- Total_Hate_Score (sum of all hate indicators)
- Hate_Label (Target Variable: 0 = Non-Hate, 1 = Hate)

---

## 🛠 Tools & Technologies Used

- Python (Pandas, NumPy)
- Matplotlib & Seaborn
- Scikit-learn
- MySQL Database
- Power BI Dashboard
- Jupyter Notebook

---

## 🔎 Step 1: Data Cleaning

- Loaded CSV dataset
- Verified data types and null values
- Converted boolean hate indicators into numeric format (0/1)
- Prepared clean dataset for analysis and modeling

---

## 📈 Step 2: Exploratory Data Analysis

Performed:
- Distribution analysis of different hate categories
- Correlation heatmap between hate indicators
- Total Hate Score distribution analysis
- Relationship analysis between toxic and abusive comments
- Identification of dominant hate speech patterns

---

## 🤖 Step 3: Predictive Modeling

### Logistic Regression

Target Variable:  
Hate_Label (0 = Non-Hate, 1 = Hate)

Evaluation Metrics:
- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC Curve & AUC Score

Objective:  
Classify comments into hate or non-hate categories for automated moderation.

---

## 🗄 Step 4: Database Integration

- Designed MySQL database schema for hate speech dataset
- Created table structure matching dataset columns
- Inserted processed CSV data into MySQL database
- Enabled SQL-based querying for hate speech analysis

---

## 📊 Step 5: Power BI Dashboard

Created an interactive dashboard to visualize hate speech insights, including:

- Distribution of different hate categories
- Hate vs Non-Hate comment comparison
- Correlation between hate indicators
- Visualization of Total Hate Score severity
- Insights for monitoring toxic content trends

Dashboard File:  
Located in `/dashboard/Hate_speech.pbix`

---

## 📌 Key Insights

- Automated hate speech detection helps social media platforms proactively filter toxic comments, improving user safety and engagement.  
- High co-occurrence of abusive, racist, and religious hate content indicates the need for multi-category moderation strategies.  
- The Total Hate Score enables prioritization of highly offensive comments for faster review and action.  
- Binary classification of comments into hate vs non-hate supports real-time automated content moderation systems.  
- Data-driven hate speech analysis assists organizations in strengthening community guidelines and compliance monitoring.

---

## 🚀 Future Enhancements

- Deploy the model for real-time comment monitoring
- Integrate live social media streaming APIs
- Implement advanced NLP models (LSTM / BERT) for improved accuracy
- Build an automated moderation alert system

---

## 📂 Project Structure

Automated-Hate-Speech-Detection/
│
├── data/                # Dataset CSV file
├── notebooks/           # Jupyter Notebook for analysis & ML model
├── database/            # SQL script for database creation
├── dashboard/           # Power BI dashboard file
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

---

## ▶ How to Run the Project

1. Clone the repository  
2. Install required libraries:  
   pip install -r requirements.txt  
3. Open notebooks/hate_speech_analysis.ipynb  
4. Run all notebook cells  
5. Execute database/create_database.sql in MySQL  
6. Open dashboard/Hate_speech_analysis.pbix in Power BI
```
