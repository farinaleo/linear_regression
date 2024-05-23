import pandas as pd


def normaliseDf(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Normalise a dataframe to obtain a μ = 0 and σ = 1 with the Z-score.
    :param df: The dataframe to normalise.
    :param cols: The columns to normalise.
    :return: The normalised dataframe.
    """
    if cols is None:
        cols = ['km', 'price']

    df_cp = df.copy(deep=True)

    for col in cols:
        _min = df_cp[col].min()
        _max = df_cp[col].max()
        df_cp[col] = df_cp[col].apply(lambda x: (x - df[col].mean()) / (df[col].std()))

    return df_cp


def denormThetas(thetas: list[float], df: pd.DataFrame) -> list[float]:
    """
    Denormalise thetas trained on a normalised dataframe.
    :param thetas: normalised thetas.
    :param df: Original dataframe.
    :return: Denormalised thetas.
    """
    t1 = thetas[1] * (df['price'].std() / df['km'].std())
    t0 = df['price'].mean() + df['price'].std() * (thetas[0] - (thetas[1] * (df['km'].mean() / df['km'].std())))

    return [t0, t1]
