import pandas as pd
from .plotModel import plotModel
from .computeTmpThetas import computeGradient0, computeGradient1


def trainModel(df: pd.DataFrame, epoch: int = 500, learningRate: float = 0.5, plot: bool = False) -> list:
    """
    Trains the model using the gradient descent algorithm.
    :param df: training data (normalised).
    :param epoch: number of epochs.
    :param learningRate: learning rate.
    :param plot: whether to plot the training process or not.
    :return: optimised model parameters (normalised).
    """
    theta0 = 0
    theta1 = 0
    savedThetas = [(theta0, theta1)]

    for e in range(epoch):
        _g0 = computeGradient0(df, (theta0, theta1), learningRate=learningRate)
        _g1 = computeGradient1(df, (theta0, theta1), learningRate=learningRate)
        theta0 -= _g0
        theta1 -= _g1
        savedThetas.append((theta0, theta1))

    if plot:
        plotModel(df, savedThetas)

    return [theta0, theta1]
