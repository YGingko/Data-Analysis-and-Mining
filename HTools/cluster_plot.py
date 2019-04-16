# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hai
@time: 2019/4/16 16:52
@desc:

"""


def cluster_plot(data, cluster_centers):
    import numpy as np
    import matplotlib.pyplot as plt

    cols = data.columns
    k = len(cluster_centers)
    color = ['b', 'g', 'r', 'c', 'y']
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)

    plot_data = np.concatenate((cluster_centers, cluster_centers[:, [0]]), axis=1)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for i in range(len(plot_data)):
        ax.plot(angles, plot_data[i], 'o-', color=color[i], label=cols[i], linewidth=2)
    ax.set_rgrids(np.arange(0.01, 3.5, 0.5), np.arange(-1, 2.5, 0.5), fontproperties="SimHei")
    ax.set_thetagrids(angles * 180 / np.pi, cols, fontproperties="SimHei")
    plt.legend(loc=4)
    return plt
