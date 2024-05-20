import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from .computeTmpThetas import computeTmpTheta0, computeTmpTheta1


def trainModel(df: pd.DataFrame, epoch: int = 500, learningRate: float = 0.5, plot: bool=False) -> list:
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

    return [theta0, theta1] if not plot else [theta0, theta1, savedThetas]
