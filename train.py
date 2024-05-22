import argparse
from srcs.extractData import extractData
from srcs.normalise import normaliseDf
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


def main():
    args = options_parser().parse_args()

    df = extractData(args.file)
    normDf = normaliseDf(df, ['km', 'price'])
    thetas = trainModel(normDf, epoch=1000, learningRate=0.1, plot=args.plot)
    # un norm thetas
    # save thetas
    print(thetas)



if __name__ == '__main__':
    main()
