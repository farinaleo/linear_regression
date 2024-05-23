import os
import json


def estimatedPrice(mileage: int, thetas: tuple = None, srcJson: str = 'thetas.json') -> float:
    if not os.path.exists(srcJson):
        raise ValueError(f'srcJson does not exist: {srcJson}')
    try:
        if thetas:
            return thetas[0] + mileage * thetas[1]
        else:
            with open(srcJson, 'r') as file:
                thetasJson = json.load(file)
                print(thetasJson)
                return float(thetasJson['theta0']) + mileage * float(thetasJson['theta1'])
    except Exception:
        return 0
