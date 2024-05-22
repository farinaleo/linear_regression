import pandas as pd
from .plotModel import plotModel
from .computeTmpThetas import computeTmpTheta0, computeTmpTheta1


def trainModel(df: pd.DataFrame, epoch: int = 500, learningRate: float = 0.5, plot: bool = False) -> list:
    theta0 = 0
    theta1 = 0
    savedThetas = [(theta0, theta1)]

    for e in range(epoch):
        _theta0 = theta0
        _theta1 = theta1
        _tmpTheta0 = computeTmpTheta0(df, (_theta0, _theta1), learningRate=learningRate)
        _tmpTheta1 = computeTmpTheta1(df, (_theta0, _theta1), learningRate=learningRate)
        theta0 -= _tmpTheta0
        theta1 -= _tmpTheta1
        savedThetas.append((theta0, theta1))

    if plot:
        plotModel(df, savedThetas)

    return [theta0, theta1]
