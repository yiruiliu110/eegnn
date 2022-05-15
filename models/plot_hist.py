import pickle
import sys

import numpy as np
import torch
import os
import yaml
from matplotlib import pyplot as plt

from estimation.graph_model import BNPGraphModel


def plot_hist(estimated_graph, data_name: str = 'TEXAS'):


    virtual_graph = estimated_graph.compute_mean_z(1000)

    z = virtual_graph._values().tolist()
    z = [item for item in z if item > 1.0]
    z = np.array(z)
    bins = np.arange(1, z.max() + 1.5, 0.1) - 0.5
    fig, ax = plt.subplots()
    ax.hist(z, bins=100, log=True)
    #ax.set_xticks(list(range(1, int(z.max()) + 1)))
    # ax.set_yticks(list(range(0, int(np.histogram(z)[0].max()) + 1)))
    fig.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figs', data_name + '_hist'))
    plt.show()
    sys.exit()
