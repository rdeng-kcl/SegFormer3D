import os
import sys
import random
from torchinfo import summary

sys.path.append("../../../")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module") # ignore monai warnings (not working)

import yaml
import torch
import argparse
import numpy as np
from typing import Dict
from termcolor import colored
from accelerate import Accelerator
from losses.losses import build_loss_fn
from optimizers.optimizers import build_optimizer
from optimizers.schedulers import build_scheduler
from train_scripts.trainer_ddp import Segmentation_Trainer
from architectures.build_architecture import build_architecture
from dataloaders.build_dataset import build_dataset, build_dataloader
from matplotlib import pyplot as plt

sys.path.append("../../brats_2017/template_experiment")
from run_experiment import (
    load_config,
    seed_everything,
    build_directories,
    display_info,
)


##################################################################################################
def launch_transfer_learning(config_path) -> Dict:
    """
    Builds Experiment
    Args:
        config (Dict): configuration file

    Returns:
        Dict: _description_
    """
    # load config
    config = load_config(config_path)

    # set seed
    seed_everything(config)

    # build directories
    # don't raise checkpoint override for development
    try:
        build_directories(config)
    except ValueError:
        warnings.warn("Checkpoint directory already exists. Overwriting...")

    # build training dataset & training data loader
    trainset = build_dataset(
        dataset_type=config["dataset_parameters"]["dataset_type"],
        dataset_args=config["dataset_parameters"]["train_dataset_args"],
    )
    trainloader = build_dataloader(
        dataset=trainset,
        dataloader_args=config["dataset_parameters"]["train_dataloader_args"],
        config=config,
        train=True,
    )

    # build validation dataset & validataion data loader
    valset = build_dataset(
        dataset_type=config["dataset_parameters"]["dataset_type"],
        dataset_args=config["dataset_parameters"]["val_dataset_args"],
    )
    valloader = build_dataloader(
        dataset=valset,
        dataloader_args=config["dataset_parameters"]["val_dataloader_args"],
        config=config,
        train=False,
    )

    # build the Model
    model = build_architecture(config)

    # load trained weights
    checkpoint_file = "C:\\Users\\Ryan Deng\\SegFormer3D\\experiments\\brats_2017\\test_trained_brats_2017\\auther_trained\\best_segformer3d_brats_performance.pth"
    state_dict = torch.load(checkpoint_file, map_location="cpu")

    # average the weights across the 4 channels for single channel input
    key = "segformer_encoder.embed_1.patch_embeddings.weight"
    assert key in state_dict
    state_dict[key] = state_dict[key].mean(dim=1, keepdim=True)

    # remove decoder weights
    for key in list(state_dict.keys()):
        if "segformer_decoder" in key:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    # don't freeze we have domain shift MRI to CT
    # # freeze the encoder
    # model.segformer_encoder.requires_grad_(False)
    # assert not any(p.requires_grad for p in model.segformer_encoder.parameters())
    # assert any(p.requires_grad for p in model.segformer_decoder.parameters())

    # set up the loss function
    criterion = build_loss_fn(
        loss_type=config["loss_fn"]["loss_type"],
        loss_args=config["loss_fn"]["loss_args"],
    )

    # set up the optimizer
    optimizer = build_optimizer(
        model=model,
        optimizer_type=config["optimizer"]["optimizer_type"],
        optimizer_args=config["optimizer"]["optimizer_args"],
    )

    # set up schedulers
    warmup_scheduler = build_scheduler(
        optimizer=optimizer, scheduler_type="warmup_scheduler", config=config
    )
    training_scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="training_scheduler",
        config=config,
    )

    # use accelarate
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=config["training_parameters"][
            "grad_accumulate_steps"
        ],
    )
    accelerator.init_trackers(
        project_name=config["project"],
        config=config,
        init_kwargs={"wandb": config["wandb_parameters"]},
    )

    # display experiment info
    display_info(config, accelerator, trainset, valset, model)

    # convert all components to accelerate
    model = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    trainloader = accelerator.prepare_data_loader(data_loader=trainloader)
    valloader = accelerator.prepare_data_loader(data_loader=valloader)
    warmup_scheduler = accelerator.prepare_scheduler(scheduler=warmup_scheduler)
    training_scheduler = accelerator.prepare_scheduler(scheduler=training_scheduler)

    # create a single dict to hold all parameters
    storage = {
        "model": model,
        "trainloader": trainloader,
        "valloader": valloader,
        "criterion": criterion,
        "optimizer": optimizer,
        "warmup_scheduler": warmup_scheduler,
        "training_scheduler": training_scheduler,
    }

    # set up trainer
    trainer = Segmentation_Trainer(
        config=config,
        model=storage["model"],
        optimizer=storage["optimizer"],
        criterion=storage["criterion"],
        train_dataloader=storage["trainloader"],
        val_dataloader=storage["valloader"],
        warmup_scheduler=storage["warmup_scheduler"],
        training_scheduler=storage["training_scheduler"],
        accelerator=accelerator,
    )

    # run train
    trainer.train()


##################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to yaml config file"
    )
    args = parser.parse_args()
    launch_transfer_learning(args.config)