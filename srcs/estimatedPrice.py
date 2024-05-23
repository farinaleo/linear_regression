import os
import json


def estimatedPrice(mileage: int, thetas: tuple = None, srcJson: str = 'thetas.json') -> float:
    """
    Estimates the price of a given car based on its mileage. The function uses given thetas or thetas from a given json file.

    :param mileage: The mileage of the car.
    :param thetas: Thetas for the equation (y = θ0 + θ1 * x).
    :param srcJson: The json file.
    :return: the supposed price (or 0 in case of error).
    """
    if not os.path.exists(srcJson) and thetas is None:
        return 0
    try:
        if thetas:
            return thetas[0] + mileage * thetas[1]
        else:
            with open(srcJson, 'r') as file:
                thetasJson = json.load(file)
                return float(thetasJson['theta0']) + mileage * float(thetasJson['theta1'])
    except Exception:
        return 0
