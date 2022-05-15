import os
import pickle

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

from estimation.generate_edge_index_and_weight import compute_dege_index_and_weight
from estimation.graph_model import BNPGraphModel


def initial_graph(edge_index, data_name: str = 'TEXAS'):
    with torch.no_grad():
        number_of_edges = int(edge_index.size()[1])
        number_of_nodes = int(torch.max(edge_index).item()) + 1

        graph = torch.sparse_coo_tensor(edge_index, torch.ones(number_of_edges),
                                        [number_of_nodes, number_of_nodes])

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_config.yml')
        with open(filename, 'r') as yaml_f:
            configs = yaml.load(yaml_f, Loader=yaml.FullLoader)
        config = configs[data_name]
        alpha = config['alpha']
        gamma = config['gamma']

        estimated_graph = BNPGraphModel(graph, alpha=alpha, tau=1.0, gamma=gamma, sigma=0.5, initial_K=10, max_K=200)

    try:
        with open(os.path.join('data', data_name, "trained_model_state_" + data_name), "rb") as output_file:
            estimated_graph.state = pickle.load(output_file)
            estimated_graph.active_K = estimated_graph.state['active_K']

    except:
        training_epochs = 50000
        result_log_likelihood = []
        results_active_K = []
        for i in range(training_epochs):
            estimated_graph.one_step(print_likelihood=True)
            result_log_likelihood.append(float(estimated_graph.log_likelihood()))
            results_active_K.append(estimated_graph.state['active_K'])

        z = estimated_graph.state['z']._values().tolist()
        z = [item for item in z if item > 1]
        z = np.array(z)
        bins = np.arange(1, z.max() + 1.5) - 0.5
        fig, ax = plt.subplots()
        ax.hist(z, bins=bins)
        ax.set_xticks(list(range(1, int(z.max())+1)))
        #ax.set_yticks(list(range(0, int(np.histogram(z)[0].max()) + 1)))
        fig.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figs', data_name + '_hist'))
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(list(range(1, training_epochs+1)), result_log_likelihood)
        ax.set_xlabel("iterations")
        ax.set_ylabel("log likelihood")
        fig.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figs', data_name + '_log_likelihood'))
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(list(range(1, training_epochs+1)), results_active_K)
        ax.set_xlabel("iterations")
        ax.set_ylabel("number of clusters")
        fig.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figs', data_name + '_active_k'))
        plt.show()





        with open(os.path.join('data', data_name, "trained_model_state_" + data_name), "wb") as output_file:
            pickle.dump(estimated_graph.state, output_file)

    virtual_graph = estimated_graph.compute_mean_z(100)

    virtual_graph = virtual_graph.to_dense()
    virtual_graph = (virtual_graph + torch.transpose(virtual_graph, 0, 1)) / 2
    virtual_graph = virtual_graph.to_sparse()

    return virtual_graph