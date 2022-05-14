import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from ray import tune

from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.trial import ExportFormat
from torch_geometric.datasets import Planetoid, WebKB
from estimation.graph_model import BNPGraphModel


def main_hpo(data_name='Cora',
             initial_K: int = 10,
             max_K: int = 100):

    # data
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_name)
    print(path)
    if data_name in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(root=path, name=data_name)
    elif data_name in ["TEXAS", "WISCONSIN", "CORNELL"]:
        data = WebKB(path, data_name)

    data = data[0]

    number_of_nodes = data.x.size()[0]
    number_of_edges = data.edge_index.size()[1]
    graph = torch.sparse_coo_tensor(data.edge_index, torch.ones(number_of_edges), [number_of_nodes, number_of_nodes])

    def train_bnp_graph_model(config, checkpoint_dir=None):
        # Create our data loaders, model, and optmizer.
        step = 0

        model = BNPGraphModel(graph, alpha=config['alpha'], tau=1.0, gamma=config['gamma'], sigma=0.5, initial_K=initial_K, max_K=max_K, if_print=False)

        # If checkpoint_dir is not None, then we are resuming from a checkpoint.
        # Load model state and iteration step from checkpoint.
        if checkpoint_dir:
            print("Loading from checkpoint.")
            path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(path)

            model.state = checkpoint["state"]
            model.active_K = model.state["active_K"]

            step = checkpoint["step"]

        while True:
            model.fit(1, print_likelihood=False)
            log_likelihood = float(model.log_likelihood())
            if step % 1 == 0:
                # Every 5 steps, checkpoint our current state.
                # First get the checkpoint directory from tune.
                with tune.checkpoint_dir(step=step) as checkpoint_dir:
                    # Then create a checkpoint file in this directory.
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    # Save state to checkpoint file.
                    # No need to save optimizer for SGD.
                    print(model.state['active_K'])
                    torch.save(
                        {
                            "step": step,
                            "state": model.state,
                            "log_likelihood": log_likelihood,
                        },
                        path,
                    )
            step += 1
            tune.report(log_likelihood=log_likelihood)

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations={
            # distribution for resampling
            "alpha": lambda: np.random.uniform(1.0, 100.0),
            # allow perturbations within this set of categorical values
            "gamma": lambda: np.random.uniform(1.0, 10.0),
        },
    )

    class CustomStopper(tune.Stopper):
        def __init__(self):
            self.should_stop = False

        def __call__(self, trial_id, result):
            max_iter = 10000
            return result["training_iteration"] >= max_iter

        def stop_all(self):
            return self.should_stop


    stopper = CustomStopper()

    analysis = tune.run(
        train_bnp_graph_model,
        name="graph_pbt",
        scheduler=scheduler,
        metric="log_likelihood",
        mode="max",
        verbose=1,
        stop=stopper,
        export_formats=[ExportFormat.MODEL],
        checkpoint_score_attr="log_likelihood",
        keep_checkpoints_num=4,
        num_samples=4,
        config={
            "alpha": 10.0,
            "gamma": 10.0,
        },
    )

    # Plot by wall-clock time
    dfs = analysis.fetch_trial_dataframes()
    # This plots everything on the same plot
    ax = None
    for d in dfs.values():
        ax = d.plot("training_iteration", "log_likelihood", ax=ax, legend=False)

    plt.xlabel("iterations")
    plt.ylabel("Test Accuracy")
    plt.show()
    print('best config:', analysis.get_best_config("log_likelihood"))


if __name__ == "__main__":
    main_hpo(data_name='Citeseer')
