import fire
import ray
import os
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from glia.exp import glia_digits
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


def tune_random(name, exp, num_samples=2, seed_value=None, **digit_kwargs):
    """Tune hyperparameters of any bandit experiment."""

    # ------------------------------------------------------------------------
    # Init seed
    prng = np.random.RandomState(seed_value)

    # Init ray
    if not ray.is_initialized():
        if "debug" in digit_kwargs:
            ray.init(local_mode=digit_kwargs["debug"])
        else:
            ray.init()

    # ------------------------------------------------------------------------
    # Create the train function. We do it scope to control if it
    # gets remote'ed to GPUs or not.
    try:
        if digit_kwargs["use_gpu"]:

            @ray.remote(num_gpus=0.25)
            def train(name=None, exp_func=None, config=None):
                trial = exp_func(**config)
                trial.update({"config": config, "name": name})
                return trial
        else:

            @ray.remote
            def train(name=None, exp_func=None, config=None):
                trial = exp_func(**config)
                trial.update({"config": config, "name": name})
                return trial
    except KeyError:

        @ray.remote
        def train(name=None, exp_func=None, config=None):
            trial = exp_func(**config)
            trial.update({"config": config, "name": name})
            return trial

    # ------------------------------------------------------------------------
    # Init:
    # Separate name from path
    path, name = os.path.split(name)

    # Look up the bandit run function were using in this tuning.
    exp_func = getattr(glia_digits, exp)

    # ------------------------------------------------------------------------
    # Run!
    # Setup the parallel workers
    runs = []
    for i in range(num_samples):
        # Make a new HP sample
        params = {}
        for k, v in digit_kwargs.items():
            if isinstance(v, str):
                params[k] = v
            elif isinstance(v, bool):
                params[k] = v
            elif isinstance(v, int):
                params[k] = v
            elif isinstance(v, float):
                params[k] = v
            else:
                low, high = v
                params[k] = prng.uniform(low=low, high=high)
        runs.append(train.remote(i, exp_func, params))

    trials = [ray.get(r) for r in runs]

    # ------------------------------------------------------------------------
    # Save configs and correct (full model data is dropped):
    best = get_best_trial(trials, 'correct')

    # Best trial config
    best_config = best["config"]
    best_config.update(get_best_result(trials, 'correct'))
    save_checkpoint(best_config,
                    filename=os.path.join(path, name + "_best.pkl"))

    # Sort and save the configs of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, 'correct')):
        sorted_configs[i] = trial["config"]
        sorted_configs[i].update({"correct": trial["correct"]})
    save_checkpoint(sorted_configs,
                    filename=os.path.join(path, name + "_sorted.pkl"))

    # kill ray
    ray.shutdown()

    # -
    return best, trials


if __name__ == '__main__':
    fire.Fire({'random': tune_random})
