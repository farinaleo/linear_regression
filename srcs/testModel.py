import pandas as pd
import math


def r_square(df: pd.DataFrame) -> float:
    """
    Calculates the R^2 score.
    :param df: data.
    :return: score.
    """
    mean = df['price'].mean()
    sum_mean = 0
    sum_predict = 0

    for _, row in df.iterrows():
        sum_predict += math.pow(row['price'] - row['predict'], 2)
        sum_mean += math.pow((row['price'] - mean), 2)

    return 1 - (sum_predict / sum_mean)


def mae(df: pd.DataFrame) -> float:
    """
    Calculates the MEA score.
    :param df: data.
    :return: score.
    """
    sum_predict = 0

    for _, row in df.iterrows():
        sum_predict += math.fabs(row['price'] - row['predict'])

    return (1 / len(df)) * sum_predict


def mse(df: pd.DataFrame) -> float:
    """
    Calculates the MSE score.
    :param df: data.
    :return: score.
    """
    sum_predict = 0

    for _, row in df.iterrows():
        sum_predict += math.pow((row['price'] - row['predict']), 2)

    return (1 / len(df)) * sum_predict


def rmse(df: pd.DataFrame) -> float:
    """
    Calculates the RMSE score.
    :param df: data.
    :return: score.
    """
    return math.sqrt(mse(df))


def mape(df: pd.DataFrame) -> float:
    """
    Calculates the MAPE score.
    :param df: data.
    :return: score.
    """
    sum_predict = 0

    for _, row in df.iterrows():
        sum_predict += math.fabs((row['price'] - row['predict']) / row['price'])

    return (1 / len(df)) * sum_predict * 100


def adjusted_r_square(df: pd.DataFrame) -> float:
    """
    Calculates the adjusted R^2 score.
    :param df: data.
    :return: score.
    """
    return 1 - ((1 - r_square(df) * (len(df) - 1)) / (len(df) - 2))
