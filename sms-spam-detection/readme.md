# SMS Spam Detection 🚫

This project is part of **INTERN ORBIT - 2 Month Internship (Level 2 Task)**.

---

##  Problem Statement

Develop a machine learning model to accurately classify SMS messages as **spam** or **legitimate**.

---

##  Goal

Utilize **TF-IDF** vectorization and machine learning algorithms like **Naive Bayes**, **Logistic Regression**, or **SVM** to process and analyze SMS text data. The objective is to identify spam messages based on learned patterns in textual content.

---

##  Skills Demonstrated

- Natural Language Processing (NLP)
- Feature Engineering with TF-IDF
- Text Classification using ML algorithms
- Streamlit Web App Deployment
- Clean Code Organization (Modular Python structure)

---

##  Project Structure

``` bash
sms-spam-detection/
├── app.py # Streamlit app
├── models/
│ ├── model.joblib # Trained ML model
│ └── vectorizer.joblib # TF-IDF Vectorizer
├── src/
│ └── preprocess.py # Text cleaning functions
├── data/
│ └── spam.csv # SMS Spam Collection Dataset
├── requirements.txt # Dependencies
└── README.md # This file
```

---


##  Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Joblib
- Regex

##  How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/internorbit.git
   cd internorbit/sms-spam-detection
   ```

2. Create and activate a virtual environment:
   ``` bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

4. Run the Streamlit app:
  ```bash
  streamlit run app.py
  ```

---

## Dataset
Source: Kaggle - SMS Spam Collection Dataset(https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Author
Jasmin Shaik
INTERN ORBIT Internship - Level 2 Task
