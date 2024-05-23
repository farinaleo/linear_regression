import argparse
import json
from srcs.extractData import extractData
from srcs.normalise import normaliseDf, denormThetas
from srcs.trainModel import trainModel


def options_parser():
    """Use to handle program parameters and options.
    """
    parser = argparse.ArgumentParser(
        prog='Train',
        description='Train a linear regression model',
        epilog='Please read the subject before proceeding to understand the input file format.')
    parser.add_argument('-f', '--file', type=str, default='data/data.csv', help='the input data file to learn')
    parser.add_argument('-p', '--plot', action='store_true', help='show the training')
    return parser


def saveThetas(thetas: list[float], fileName: str = 'thetas.json'):
    """
    Save the trained model in a json file.
    :param thetas: model.
    :param fileName: json file.
    :return:
    """
    try:
        data = {'theta0': str(thetas[0]), 'theta1': str(thetas[1])}

        with open(fileName, 'w') as file:
            json.dump(data, file, indent=4)

    except Exception:
        raise ValueError('Impossible to save thetas')


def main():
    args = options_parser().parse_args()

    print("Loading data...")
    df = extractData(args.file)

    print("Normalising data...")
    normDf = normaliseDf(df, ['km', 'price'])

    print("Training model...")
    thetas = trainModel(normDf, epoch=1000, learningRate=0.1, plot=args.plot)

    print("Denormalising model...")
    thetas = denormThetas(thetas, df)

    print("Saving model...")
    saveThetas(thetas)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('The programme has just been stopped suddenly')
        print(f'Got: {e}')
