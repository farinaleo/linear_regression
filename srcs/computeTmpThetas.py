import pandas as pd
from .estimatedPrice import estimatedPrice


def computeTmpTheta0(df: pd.DataFrame, thetas: tuple, learningRate: float = 0.5) -> float:
    tmpSum = 0
    for _, row in df.iterrows():
        tmpSum += estimatedPrice(row['km'], thetas=thetas) - row['price']
    return learningRate / len(df) * tmpSum


def computeTmpTheta1(df: pd.DataFrame, thetas: tuple, learningRate: float = 0.5) -> float:
    tmpSum = 0
    for _, row in df.iterrows():
        tmpSum += (estimatedPrice(row['km'], thetas=thetas) - row['price']) * row['km']
    return learningRate / len(df) * tmpSum
