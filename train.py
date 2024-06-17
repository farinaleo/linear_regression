import argparse
import json
from srcs.extractData import extract_data
from srcs.normalise import normalise_df, denorm_thetas
from srcs.trainModel import train_model


def options_parser():
    """Use to handle program parameters and options.
    """
    parser = argparse.ArgumentParser(
        prog='Train',
        description='Train a linear regression model',
        epilog='Please read the subject before proceeding to understand the input file format.')
    parser.add_argument('-f', '--file', type=str, default='data/data.csv', help='the input data file to learn')
    parser.add_argument('-e', '--epoch', type=int, default=1000, help='iteration number for training')
    parser.add_argument('-l', '--learningRate', type=float, default=0.1, help='learning rate for training model')
    parser.add_argument('-p', '--plot', action='store_true', help='show the training')
    return parser


def save_thetas(thetas: list[float], file_name: str = 'thetas.json'):
    """
    Save the trained model in a json file.
    :param thetas: model.
    :param file_name: json file.
    :return:
    """
    try:
        data = {'theta0': str(thetas[0]), 'theta1': str(thetas[1])}

        with open(file_name, 'w') as file:
            json.dump(data, file, indent=4)

    except Exception:
        raise ValueError('Impossible to save thetas')


def main():
    args = options_parser().parse_args()

    print("Loading data...")
    df = extract_data(args.file)

    print("Normalising data...")
    normDf = normalise_df(df, ['km', 'price'])

    print("Training model...")
    thetas = train_model(normDf, epoch=args.epoch, learning_rate=args.learningRate, plot=args.plot)

    print("Denormalising model...")
    thetas = denorm_thetas(thetas, df)

    print("Saving model...")
    save_thetas(thetas)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('The programme has just been stopped suddenly')
        print(f'Got: {e}')
