import os
import json


def estimated_price(mileage: int, thetas: tuple = None, src_json: str = 'thetas.json') -> float:
    """
    Estimates the price of a given car based on its mileage. The function uses given thetas or thetas from a given
    json file.

    :param mileage: The mileage of the car.
    :param thetas: Thetas for the equation (y = θ0 + θ1 * x).
    :param src_json: The json file.
    :return: the supposed price (or 0 in case of error).
    """
    if not os.path.exists(src_json) and thetas is None:
        return 0
    try:
        if thetas:
            return thetas[0] + mileage * thetas[1]
        else:
            with open(src_json, 'r') as file:
                thetas_json = json.load(file)
                return float(thetas_json['theta0']) + mileage * float(thetas_json['theta1'])
    except Exception:
        return 0
