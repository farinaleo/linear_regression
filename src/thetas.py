#   ------------------------------------------------------------------------------------------------------------------ #
#   contact : leo.farina.fr@gmail.com                                                                 ░▄▄▄▄░           |
#   github : https://github.com/farinaleo                                                             ▀▀▄██►           |
#   date : 3/17/24, 4:33 PM                                                                           ▀▀███►           |
#                                                                                                     ░▀███►░█►        |
#                                                                                                     ▒▄████▀▀         |
#   ------------------------------------------------------------------------------------------------------------------ #
#  Copyright (c) 2024.
import json

from typing import Tuple

thetas_file = 'datas/thetas.json'
default_json = {
    'theta_0': 0.0,
    'theta_1': 0.0
}


def get_thetas() -> Tuple[float, float]:
    """ Get thetas value from the json file.
    If the file doesn't exist it will be created.
    :return: Tuple of thetas as (theta 0, theta 1)
    """
    try:
        with open(thetas_file, 'r') as f:
            data = json.load(f)
            return data['theta_0'], data['theta_1']
    except Exception:
        create_thetas(default_json)
        return default_json['theta_0'], default_json['theta_1']


def edit_theta0(value: float):
    """ Change theta 0 value
    :param value: New theta value
    """
    try:
        with open(thetas_file, 'r+') as file:
            data = json.load(file)
            data['theta_0'] = value
            file.seek(0)
            json.dump(data, file)
            file.truncate()
    except Exception:
        cp_default = default_json
        cp_default['theta_0'] = value
        create_thetas(cp_default)


def edit_theta1(value: float):
    """ Change theta 1 value
    :param value: New theta value
    """
    try:
        with open(thetas_file, 'r+') as file:
            data = json.load(file)
            data['theta_1'] = value
            file.seek(0)
            json.dump(data, file)
            file.truncate()
    except Exception:
        cp_default = default_json
        cp_default['theta_1'] = value
        create_thetas(cp_default)


def create_thetas(config_json):
    """ Create thetas json file with the default values
    """
    try:
        with open(thetas_file, 'w') as file:
            json.dump(config_json, file)
    except Exception:
        print('Could not create the config file')


if __name__ == '__main__':
    print(get_thetas())
    edit_theta0(get_thetas()[0] * 3)
    edit_theta1(get_thetas()[1] * 4)
    print(get_thetas())
