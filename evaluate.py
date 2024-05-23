import json
from srcs.extractData import extractData
from srcs.testModel import rSquare, rmse, mae, mse, mape, adjustedRSquare


def main():
    df = extractData('data/data.csv')
    thetas = ()
    with open('thetas.json', 'r') as file:
        jsonfile = json.load(file)
        thetas = [float(jsonfile['theta0']), float(jsonfile['theta1'])]

    df['predict'] = df['km'].apply(lambda x: thetas[0] + thetas[1] * x)
    print(f'MAE\t:\t{mae(df)}')
    print(f'MSE\t:\t{mse(df)}')
    print(f'RMSE\t:\t{rmse(df)}')
    print(f'MAPE\t:\t{mape(df)}')
    print(f'R2\t:\t{rSquare(df)}')
    print(f'AdjR2\t:\t{adjustedRSquare(df)}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('An error occurred, SO SO SORRY !!!!')
        print(f'error : {e}')
