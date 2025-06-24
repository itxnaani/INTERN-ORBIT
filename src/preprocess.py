import pandas as pd

def load_and_preprocess(filepath):
    
    df = pd.read_csv(filepath, encoding='latin1')
    # Keep only relevant columns
    df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']].dropna()

    # Combine actor names and genre into one feature string
    df['features'] = (
        df['Genre'].astype(str) + ' ' +
        df['Director'].astype(str) + ' ' +
        df['Actor 1'].astype(str) + ' ' +
        df['Actor 2'].astype(str) + ' ' +
        df['Actor 3'].astype(str)
    )

    return df




