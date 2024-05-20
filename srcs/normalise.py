import pandas as pd


def normaliseDf(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = ['km', 'price']

    df_cp = df.copy(deep=True)

    for col in cols:
        _min = df_cp[col].min()
        _max = df_cp[col].max()
        df_cp[col] = df_cp[col].apply(lambda x: (x - _min) / (_max - _min))

    return df_cp
