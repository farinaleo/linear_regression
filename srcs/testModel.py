import pandas as pd
import math


def rSquare(df: pd.DataFrame) -> float:
    """
    Calculates the R^2 score.
    :param df: data.
    :return: score.
    """
    mean = df['price'].mean()
    sumMean = 0
    sumPredict = 0

    for _, row in df.iterrows():
        sumPredict += math.pow(row['price'] - row['predict'], 2)
        sumMean += math.pow((row['price'] - mean), 2)

    return 1 - (sumPredict / sumMean)


def mae(df: pd.DataFrame) -> float:
    """
    Calculates the MEA score.
    :param df: data.
    :return: score.
    """
    sumPredict = 0

    for _, row in df.iterrows():
        sumPredict += math.fabs(row['price'] - row['predict'])

    return (1 / len(df)) * sumPredict


def mse(df: pd.DataFrame) -> float:
    """
    Calculates the MSE score.
    :param df: data.
    :return: score.
    """
    sumPredict = 0

    for _, row in df.iterrows():
        sumPredict += math.pow((row['price'] - row['predict']), 2)

    return (1 / len(df)) * sumPredict


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
    sumPredict = 0

    for _, row in df.iterrows():
        sumPredict += math.fabs((row['price'] - row['predict']) / row['price'])

    return (1 / len(df)) * sumPredict * 100


def adjustedRSquare(df: pd.DataFrame) -> float:
    """
    Calculates the adjusted R^2 score.
    :param df: data.
    :return: score.
    """
    return 1 - ((1 - rSquare(df) * (len(df) - 1)) / (len(df) - 2))
