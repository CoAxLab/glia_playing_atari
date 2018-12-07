"""Hyperparameters sweeps. Glia learning digits"""
import fire
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

import ray
import ray.tune as tune
import numpy as np

from glia.exp.glia_digits import train
from glia.exp.glia_digits import test
from glia.exp.glia_digits import train_vae
from glia.exp.glia_digits import test_vae
from glia.exp.glia_digits import VAE
from glia.exp.glia_digits import PerceptronGlia


def hyper_run(config, reporter):
    """Glia learn to see (digits)"""
    default = dict(
        batch_size=128,
        test_batch_size=128,
        epochs=10,
        lr=0.01,
        lr_vae=1e-3,
        use_cuda=False,
        seed=42,
        log_interval=50,
        progress=False,
        debug=False,
        data_path=None,
        use_cude=True,
    )
    default.update(config)
    config = default

    # ------------------------------------------------------------------------
    # Training settings
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    if config["use_cuda"]:
        torch.cuda.manual_seed(config["seed"])
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if config["data_path"] is None:
        config["data_path"] = "data"

    # ------------------------------------------------------------------------
    # Get and pre-process data
    kwargs = {
        'num_workers': 1,
        'pin_memory': True
    } if config["use_cuda"] else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            config["data_path"],
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=config["batch_size"],
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            config['data_path'],
            train=False,
            transform=transforms.ToTensor(),
        ),
        batch_size=config["test_batch_size"],
        shuffle=True,
        **kwargs)

    # ------------------------------------------------------------------------
    # Model init
    model_vae = VAE()
    if config["use_cuda"]:
        model_vae.cuda()
    optimizer_vae = optim.Adam(model_vae.parameters(), lr=config["lr_vae"])

    model = PerceptronGlia()
    if config["use_cuda"]:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Learn classes
    for epoch in range(1, config["epochs"] + 1):
        # Learn latent z
        train_vae(
            model_vae,
            device,
            train_loader,
            optimizer_vae,
            epoch,
            log_interval=config["log_interval"],
            debug=False)
        vae_loss = test_vae(
            model_vae,
            device,
            test_loader,
            epoch,
            config["test_batch_size"],
            debug=False,
            progress=False,
            data_path=config["data_path"])

        # Learn to decide
        train(
            model,
            model_vae,
            device,
            train_loader,
            optimizer,
            epoch,
            log_interval=config["log_interval"],
            debug=False)
        test_loss, test_correct = test(
            model, model_vae, device, test_loader, debug=False, progress=False)

        reporter(mean_loss=test_loss, mean_accuracy=test_correct)


def digit_tune_1(data_path,
                 max_iteration=100,
                 use_cuda=False,
                 num_cpus=8,
                 num_gpus=4):
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    experiment_spec = {
        "digit_tune_1": {
            "run": hyper_run,
            "stop": {
                "mean_accuracy": 0.75,
                "training_iteration": max_iteration,
            },
            "config": {
                "data_path":
                data_path,
                "use_cuda":
                use_cuda,
                "lr":
                lambda spec: np.random.uniform(0.0001, .1),
                "lr_vae":
                lambda spec: np.random.uniform(0.001, .1),
                "num_hidden":
                lambda spec: np.random.randint(1, 20),
                "activation_function":
                lambda spec: np.random.choice([
                    "ELU",
                    "ReLU",
                    "Tanh",
                    "SELU",
                    "Sigmoid", ])
            },
            "trial_resources": {
                "cpu": 1,
                "gpu": 0.5  # two models/gpu please
            },
            "num_samples": 1,
            "local_dir": data_path,
            "max_failures": 2
        }
    }
    tune.run_experiments(experiment_spec, verbose=1)


if __name__ == "__main__":
    fire.Fire({"digit_tune_1": digit_tune_1})
