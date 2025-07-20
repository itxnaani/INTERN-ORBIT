# Global Terrorism Analysis

This project is part of **INTERN ORBIT - 2 Month Internship (Level 3 Task)**.

---

##  Problem Statement

This project focuses on analyzing global terrorism data to uncover patterns, trends, and possible future threats. The goal is to extract actionable insights that can inform counter-terrorism strategies and decision-making processes.

---

##  Objective

To develop a machine learning model that can:
- Analyze terrorism data over time
- Identify key trends in terrorist tactics, organizations, and targets
- Predict potential threats and attacks based on historical data
  
---

##  Tech Stack & Tools

- Programming Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly
- ML Models: Random Forest, Logistic Regression, KMeans Clustering
- Visualization: Plotly, Seaborn
- Deployment: Streamlit (app.py)

---

##  Project Structure

``` bash
gtd_analysis/
│
├── data/                # Dataset and processed files
├── src/                 # Core scripts for data preprocessing and EDA
├── models/              # Trained models and experimentation
├── app.py               # Streamlit Web App for visualization & prediction
├── main.py              # Entry script for running pipeline
└── requirements.txt     # Required Python dependencies
```


##  Models Implemented

- Exploratory Data Analysis (EDA) on patterns and hot zones
- Clustering to identify risk regions
- Supervised learning to classify threat levels or predict possible future targets

##  How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/internorbit.git
   cd internorbit/gtd_analysis
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
Source: Global Terrorism Database (GTD) - Kaggle(https://www.kaggle.com/datasets/START-UMD/gtd?resource=download)

## Author
Jasmin Shaik
INTERN ORBIT Internship - Level 3 Task
