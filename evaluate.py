import json
from srcs.extractData import extract_data
from srcs.testModel import r_square, rmse, mae, mse, mape, adjusted_r_square


def main():
    df = extract_data('data/data.csv')
    thetas = ()
    with open('thetas.json', 'r') as file:
        jsonfile = json.load(file)
        thetas = [float(jsonfile['theta0']), float(jsonfile['theta1'])]

    df['predict'] = df['km'].apply(lambda x: thetas[0] + thetas[1] * x)
    print('Lower is better')
    print('MAE\t: {:.2f}'.format(mae(df)))
    print('MSE\t: {:.2f}'.format(mse(df)))
    print('RMSE\t: {:.2f}'.format(rmse(df)))
    print('MAPE\t: {:.2f} %'.format(mape(df)))
    print('Higher is better')
    print('R2\t: {:.2f} %'.format(r_square(df)))
    print('AdjR2\t: {:.2f}'.format(adjusted_r_square(df)))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('An error occurred, SO SO SORRY !!!!')
        print(f'error : {e}')
