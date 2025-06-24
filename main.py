from src.preprocess import load_and_preprocess
from src.model import train_model, evaluate_model

def main():
    df = load_and_preprocess("data/IMDb Movies India.csv")
    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
