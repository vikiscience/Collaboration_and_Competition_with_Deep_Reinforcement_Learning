import const

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_history_rolling_mean(hist, N=const.rolling_mean_N, fp=const.file_path_img_score, use_scatter=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # prepare data
    x = pd.Series(hist)
    y = x.rolling(window=N).mean().iloc[N - 1:]

    # plot scores for each iteration (history)
    if use_scatter:
        plt.scatter(x=range(len(hist)), y=hist, c='darkorchid', s=2, label='score')
    else:
        plt.plot(hist, c='darkorchid', marker='.', markevery=[-1], label='score')

    # plot rolling mean scores
    # plt.plot(y, c='blue', marker='.', markevery=[-1], label='rolling_score (N={})'.format(N))
    plt.plot(y, c='blue', label='rolling_score (N={})'.format(N))

    # plot line to signify the aimed high score
    x1, x2 = 0, len(hist)
    y1, y2 = const.high_score, const.high_score
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='--', linewidth=1, label='high_score')

    # annotate last_history and last_rolling point
    last_points = [(len(hist) - 1, hist[-1])]
    if not y.empty:
        last_points.append((len(hist) - 1, y.iloc[-1]))  # last rolling value
        last_points.append((y.idxmax(), y.max()))  # max rolling value

    for (i, j) in last_points:
        # ax.annotate('{:.3f}'.format(j), xy=(i, j), xytext=(i + 0.1, j))
        ax.plot([i], [j], marker='o', color='red', markersize=4)
        plt.text(i, j, '{:.3f}'.format(j))

    plt.xlabel('Episodes')
    plt.ylabel('Score (Sum of Rewards)')
    plt.title('Online Performance')
    plt.legend(loc='best')
    plt.savefig(fp)
    plt.close()


def plot_loss(loss_lists, labels,
              fp=const.file_path_img_loss):
    for i, l in enumerate(loss_lists):
        x = [elem[0] for elem in l]
        y = [elem[1] for elem in l]
        plt.plot(x, y, label=labels[i])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.title('Actor & Critic Losses')
    plt.savefig(fp)
    plt.close()


def plot_scatter(data, title_text='Text', fp=const.file_path_img_actions):
    if type(data) == list:
        data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], label='agent_1')
    if data.shape[1] == 4:
        plt.scatter(data[:, 2], data[:, 3], label='agent_2')
    plt.xlabel('move')
    plt.ylabel('jump')
    plt.legend(loc='best')
    plt.title(title_text)
    plt.savefig(fp)
    plt.close()
