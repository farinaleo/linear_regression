import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def plot_model(df: pd.DataFrame, thetas: list):
    """
    Plot the training process.
    :param df: The training data.
    :param thetas: list of thetas.
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    max_frames = len(thetas)

    df_cp = df.copy(deep=True)

    def update(i):
        """anim graph"""
        if len(thetas) > 0:
            t = thetas.pop(0)
            df_cp['test'] = df_cp['km'].apply(lambda x: t[0] + t[1] * x)
            ax.clear()
            ax.scatter(df_cp['price'], df_cp['km'])
            ax.plot(df_cp['test'], df_cp['km'], color='red')

    a = anim.FuncAnimation(fig, update, repeat=False, interval=10, frames=max_frames, cache_frame_data=False)
    plt.show()
    return a
