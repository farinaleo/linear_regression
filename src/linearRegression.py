#   ------------------------------------------------------------------------------------------------------------------ #
#   contact : leo.farina.fr@gmail.com                                                                 ░▄▄▄▄░           |
#   github : https://github.com/farinaleo                                                             ▀▀▄██►           |
#   date : 3/17/24, 3:06 PM                                                                           ▀▀███►           |
#                                                                                                     ░▀███►░█►        |
#                                                                                                     ▒▄████▀▀         |
#   ------------------------------------------------------------------------------------------------------------------ #
#  Copyright (c) 2024.

from estimatePrice import estimatePrice
from extract_csv import extract_csv
from thetas import get_thetas, edit_theta0, edit_theta1
import pandas as pd
import matplotlib.pyplot as plt


def trainLinearRegression():
    """ Determine the optimal theta 0  and theta 1 with the datas/data.csv by performing the linear regression.
    """
    datas = extract_csv()
    datas.sort_values(by=['km'])

    edit_theta0(0)
    edit_theta1(0)
    for index, row in datas.iterrows():
        optimise_thetas(datas, index, 0.007)

    print("Optimal theta")
    print(get_thetas())

    x = [x_v for x_v in range(datas['km'].min(), datas['km'].max())]
    print(f'find the price for {len(x)} cars')
    y = [estimatePrice(x_v) for x_v in x]
    print('price found')
    plt.scatter(datas['km'], datas['price'])
    plt.plot(x, y, color='red')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()

def optimise_thetas(dataset, index, learning_rate):
    """

    :param dataset: the datas from csv.
    :param index: the index to stop.
    :param learning_rate: the
    """
    i = 0
    sum0 = 0
    sum1 = 0
    while i < index + 1:
        sum0 = sum0 + compute_estimate_for_0(dataset.iloc[i]['km'], dataset.iloc[i]['price'])
        i = i + 1
    i = 0
    while i < index + 1:
        sum1 = sum1 + compute_estimate_for_1(dataset.iloc[i]['km'], dataset.iloc[i]['price'])
        i = i + 1
    theta0 = learning_rate * (1 / (index + 1)) * sum0
    theta1 = learning_rate * (1 / (index + 1)) * sum1
    edit_theta0(theta0)
    edit_theta1(theta1)


def compute_estimate_for_0(mileage, price) -> float:
    """ Compute estimatePrice(mileage[i]) − price[i]
    """
    return estimatePrice(mileage) - price


def compute_estimate_for_1(mileage, price) -> float:
    """ Compute estimatePrice(mileage[i]) − price[i]) ∗ mileage[i]
    """
    return estimatePrice(mileage - price) * mileage


if __name__ == "__main__":
    trainLinearRegression()