import pandas as pd
from .estimatedPrice import estimatedPrice


def computeGradient0(df: pd.DataFrame, thetas: tuple, learningRate: float = 0.5) -> float:
    """
    Apply the given formula from the subject to compute the actual gradiant (g0).
    :param df: Dataframe containing the subject data (normalised).
    :param thetas: Tuple of temporary thetas to improve.
    :param learningRate: Learning rate.
    :return: Gradient 0.
    """
    tmpSum = 0
    for _, row in df.iterrows():
        tmpSum += estimatedPrice(row['km'], thetas=thetas) - row['price']
    return learningRate / len(df) * tmpSum


def computeGradient1(df: pd.DataFrame, thetas: tuple, learningRate: float = 0.5) -> float:
    """
        Apply the given formula from the subject to compute the actual gradiant (g1).
        :param df: Dataframe containing the subject data (normalised).
        :param thetas: Tuple of temporary thetas to improve.
        :param learningRate: Learning rate.
        :return: Gradient 1.
        """
    tmpSum = 0
    for _, row in df.iterrows():
        tmpSum += (estimatedPrice(row['km'], thetas=thetas) - row['price']) * row['km']
    return learningRate / len(df) * tmpSum
