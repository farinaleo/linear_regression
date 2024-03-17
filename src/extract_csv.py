#   ------------------------------------------------------------------------------------------------------------------ #
#   contact : leo.farina.fr@gmail.com                                                                 ░▄▄▄▄░           |
#   github : https://github.com/farinaleo                                                             ▀▀▄██►           |
#   date : 3/17/24, 2:43 PM                                                                           ▀▀███►           |
#                                                                                                     ░▀███►░█►        |
#                                                                                                     ▒▄████▀▀         |
#   ------------------------------------------------------------------------------------------------------------------ #
#  Copyright (c) 2024.

import pandas as pd

csv_filename = 'datas/data.csv'


def extract_csv():
    """ Open the csv file, check the format and return a pandas dataframe
    :return: None in case of failure, else the corresponding dataframe.
    """
    try:
        dataframe = pd.read_csv(csv_filename)
        if set(dataframe.columns) == {"km", "price"}:
            dataframe["km"] = pd.to_numeric(dataframe["km"], errors="raise")
            dataframe["price"] = pd.to_numeric(dataframe["price"], errors="raise")
        else:
            raise ValueError("Error data format")
    except Exception as e:
        print('Could not open the data file correctly')
        print(f'Error message {e}')
        return None
    print(dataframe)
    return dataframe


if __name__ == '__main__':
    extract_csv()
