import pandas as pd
from .estimatedPrice import estimated_price


def compute_partial_derivative_0(df: pd.DataFrame, thetas: tuple) -> float:
    """
    Apply the given formula from the subject to compute the actual gradiant (g0).
    The given formula can be associate to a partial derivative (see the README.MD).
    :param df: Dataframe containing the subject data (normalised).
    :param thetas: Tuple of temporary thetas to improve.
    :return: Gradient 0.
    """
    tmp_sum = 0
    for _, row in df.iterrows():
        tmp_sum += estimated_price(row['km'], thetas=thetas) - row['price']
    return 1 / len(df) * tmp_sum


def compute_partial_derivative_1(df: pd.DataFrame, thetas: tuple) -> float:
    """
        Apply the given formula from the subject to compute the actual gradiant (g1).
    The given formula can be associate to a partial derivative (see the README.MD).

        :param df: Dataframe containing the subject data (normalised).
        :param thetas: Tuple of temporary thetas to improve.
        :return: Gradient 1.
        """
    tmp_sum = 0
    for _, row in df.iterrows():
        tmp_sum += (estimated_price(row['km'], thetas=thetas) - row['price']) * row['km']
    return 1 / len(df) * tmp_sum
