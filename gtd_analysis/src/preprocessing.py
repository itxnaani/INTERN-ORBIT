import pandas as pd

def load_data(path):
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    return df

def clean_data(df):
    df = df[['iyear', 'imonth', 'iday', 'country_txt', 'region_txt',
             'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt',
             'nkill', 'nwound', 'gname', 'summary']].copy()  # ← COPY is key

    # Proper fillna usage — no inplace
    df['nkill'] = df['nkill'].fillna(0)
    df['nwound'] = df['nwound'].fillna(0)

    # Drop NA in 'summary'
    df = df.dropna(subset=['summary'])

    return df
