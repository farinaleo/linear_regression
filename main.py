from srcs.extractData import extractData
from srcs.normalise import normaliseDf
from srcs.estimatedPrice import estimatedPrice
import matplotlib.animation as anim
from srcs.trainModel import trainModel
import matplotlib.pyplot as plt

df = extractData('data/data.csv')
normedDf = normaliseDf(df)
thetas = trainModel(normedDf, plot=True, learningRate=0.1, epoch=1000)
# print(thetas)
