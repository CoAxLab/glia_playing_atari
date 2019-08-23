import fire
import ray
import os
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from glia import exp
from glia.util import save_checkpoint
from glia.util import load_checkpoint

from functools import partial


def get_best_trial(trial_list, metric):
    """Retrieve the best trial."""
    return max(trial_list, key=lambda trial: trial[metric])


def get_sorted_trials(trial_list, metric):
    return sorted(trial_list, key=lambda trial: trial[metric], reverse=True)


def get_best_result(trial_list, metric):
    """Retrieve the last result from the best trial."""
    return {metric: get_best_trial(trial_list, metric)[metric]}


def get_configs(trial_list):
    """Extract configs"""

    return [trial["config"] for trial in trial_list]


def get_metrics(trial_list, metric):
    """Extract metric"""
    return [trial[metric] for trial in trial_list]


def train(exp_func=None, num_epochs=None, seed_value=None, config=None):

    # Run
    trial = exp_func(num_epochs=num_epochs, seed_value=seed_value, **config)

    # Save metadata
    trial.update({
        "config": config,
        "num_epochs": num_epochs,
        "seed_value": seed_value
    })

    return trial


def tune_random(name,
                exp_name,
                num_epochs=1000,
                batch_size=128,
                test_batch_size=128,
                glia=False,
                num_samples=10,
                num_processes=1,
                verbose=False,
                seed_value=None,
                config=None,
                **sample_kwargs):
    """Tune hyperparameters of any bandit experiment."""
    prng = np.random.RandomState(seed_value)

    # ------------------------------------------------------------------------
    # Init:
    # Separate name from path
    path, name = os.path.split(name)

    # Look up the bandit run function were using in this tuning.
    exp_func = getattr(exp, exp_name)

    # Build the parallel callback
    trials = []

    def append_to_results(result):
        trials.append(result)

    # Setup default params
    params = dict(
        exp_func=exp_func,
        num_epochs=num_epochs,
        seed_value=seed_value,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        config={})

    # ------------------------------------------------------------------------
    # Run!
    # Setup the parallel workers
    workers = []
    pool = Pool(processes=num_processes)
    for _ in range(num_samples):

        # Reset param sample for safety
        if config is None:
            params["config"] = {}
        else:
            params["config"] = config

        # Make a new sample
        for k, (low, high) in sample_kwargs.items():
            params["config"][k] = prng.uniform(low=low, high=high)

        # A worker gets the new sample
        workers.append(
            pool.apply_async(
                train, kwds=deepcopy(params), callback=append_to_results))

    # Get the worker's result (blocks until complete)
    for worker in workers:
        worker.get()
    pool.close()
    pool.join()

    # ------------------------------------------------------------------------
    # Save configs and total_R (full model data is dropped):
    best = get_best_trial(trials, 'total_R')

    # Best trial config
    best_config = best["config"]
    best_config.update(get_best_result(trials, 'total_R'))
    save_checkpoint(
        best_config, filename=os.path.join(path, name + "_best.pkl"))

    # Sort and save the configs of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, 'total_R')):
        sorted_configs[i] = trial["config"]
        sorted_configs[i].update({"total_R": trial["total_R"]})
    save_checkpoint(
        sorted_configs, filename=os.path.join(path, name + "_sorted.pkl"))

    return best, trials
