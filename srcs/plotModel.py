import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def plotModel(df: pd.DataFrame, thetas: list):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    maxFrames = len(thetas)

    dfCp = df.copy(deep=True)

    def update(i):
        if len(thetas) > 0:
            t = thetas.pop(0)
            dfCp['test'] = dfCp['km'].apply(lambda x: t[0] + t[1] * x)
            ax.clear()
            ax.scatter(dfCp['price'], dfCp['km'])
            ax.plot(dfCp['test'], dfCp['km'], color='red')

    a = anim.FuncAnimation(fig, update, repeat=False, interval=10, frames=maxFrames, cache_frame_data=False)
    plt.show()
    return a
