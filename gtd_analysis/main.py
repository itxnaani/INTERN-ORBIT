import pandas as pd
import joblib
from src.preprocessing import load_data, clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = load_data('data/globalterrorismdb_0718dist.csv')
df = clean_data(df)

# Target: Terrorist group (you can also do classification on weapon/attack type)
df['label'] = df['gname'].apply(lambda x: 1 if x != 'Unknown' else 0)

X = df['summary']
y = df['label']

vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'models/terror_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
