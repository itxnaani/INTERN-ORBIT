from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import Ridge


def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['features'])
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Save model and vectorizer for later use
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
