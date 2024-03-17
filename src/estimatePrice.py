#   ------------------------------------------------------------------------------------------------------------------ #
#   contact : leo.farina.fr@gmail.com                                                                 ░▄▄▄▄░           |
#   github : https://github.com/farinaleo                                                             ▀▀▄██►           |
#   date : 3/17/24, 3:05 PM                                                                           ▀▀███►           |
#                                                                                                     ░▀███►░█►        |
#                                                                                                     ▒▄████▀▀         |
#   ------------------------------------------------------------------------------------------------------------------ #
#  Copyright (c) 2024.

from thetas import get_thetas


def estimatePrice(mileage) -> float:
    """ Estimate the price of mileage.
    :param mileage: the car mileage.
    :return: the estimated price.
    """
    theta0, theta1 = get_thetas()
    return theta0 + (theta1 * mileage)


if __name__ == "__main__":
    mileage_insert = None
    got_mileage = False
    while not got_mileage:
        try:
            mileage_insert = float(input("What is the mileage? : "))
            if type(mileage_insert) is float and mileage_insert >= 0:
                got_mileage = True
            else:
                raise ValueError('Invalid input')
        except ValueError as e:
            print("Please insert a correct mileage !!!!")
            print(f'{e}')
    print(f'The estimated price is {estimatePrice(mileage_insert)}')
