import os
import pandas as pd


def extract_data(src: str, sep: str = ',', head=None) -> pd.DataFrame:
    """
    Extract data from a csv file and clean it.
    :param src: the path to the csv file.
    :param sep: delimiter used to separate data in csv.
    :param head: required column name.
    :return: data as dataframe.
    """
    if head is None:
        head = ['km', 'price']
    if not os.path.isfile(src) or not src.endswith('.csv'):
        raise ValueError(f'File {src} not found or not a .csv file.')

    try:
        df = pd.read_csv(src, sep=sep, dtype=float)
        for h in df.head():
            if h not in head:
                raise ValueError(f'Columns {head} are not present in the dataframe.')
        return df
    except Exception as e:
        raise e
