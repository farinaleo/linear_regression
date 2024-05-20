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

testdf = normedDf.copy(deep=True)
# testdf['test'] = testdf['km'].apply(lambda x: thetas[0] + thetas[1] * x)
haaaha: list = list(thetas[2])

y = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


def update(i):
    t = haaaha.pop(0)
    testdf['test'] = testdf['km'].apply(lambda x: t[0] + t[1] * x)
    ax.clear()
    ax.scatter(testdf['price'], testdf['km'])
    ax.plot(testdf['test'], testdf['km'], color='red')


a = anim.FuncAnimation(fig, update, repeat=False, interval=10)
# plt.show()
a.save('animation.mp4')