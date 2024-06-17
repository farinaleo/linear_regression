import pandas as pd
from .plotModel import plot_model
from .computeTmpThetas import compute_partial_derivative_0, compute_partial_derivative_1
from tqdm import tqdm


def train_model(df: pd.DataFrame, epoch: int = 500, learning_rate: float = 0.5, plot: bool = False) -> list:
    """
    Trains the model using the gradient descent algorithm.
    :param df: training data (normalised).
    :param epoch: number of epochs.
    :param learning_rate: learning rate.
    :param plot: whether to plot the training process or not.
    :return: optimised model parameters (normalised).
    """
    theta0 = 0
    theta1 = 0
    saved_thetas = [(theta0, theta1)]

    for _ in tqdm(range(epoch), desc="Progress", ncols=100, colour='blue'):
        _t0_tmp = compute_partial_derivative_0(df, (theta0, theta1), learning_rate=learning_rate)
        _t1_tmp = compute_partial_derivative_1(df, (theta0, theta1), learning_rate=learning_rate)
        theta0 -= _t0_tmp
        theta1 -= _t1_tmp
        saved_thetas.append((theta0, theta1))

    if plot:
        plot_model(df, saved_thetas)

    return [theta0, theta1]
