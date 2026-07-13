import sys
from torchinfo import summary

sys.path.append("../../../")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module") # ignore monai warnings (not working)

import torch
import numpy as np
from architectures.build_architecture import build_architecture
from dataloaders.build_dataset import build_dataset, build_dataloader
from matplotlib import pyplot as plt

sys.path.append("../template_experiment")
from run_experiment import load_config


if __name__ == '__main__':

    config = load_config("config.yaml")

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

    cuda = "cuda:0"

    model = build_architecture(config)

    # checkpoint_file = "./model_checkpoints/best_dice_checkpoint/pytorch_model.bin"
    checkpoint_file = "./model_checkpoints/auther_trained/best_segformer3d_brats_performance.pth"
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state_dict)

    model = model.to(cuda)
    model = model.eval()

    summary(model, input_size=(1, 4, 96, 96, 96), row_settings=["var_names"])
    exit()

    with torch.no_grad():
        raw_data = next(iter(valloader))
        data, labels = (
            raw_data["image"],
            raw_data["label"],
        )

        predicted = model(data.to(cuda)).detach().cpu().numpy()
        data = data.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    s = np.s_[0, 0, 64, :, :]

    # plt.figure(); plt.imshow(data[s])
    # plt.figure(); plt.imshow(labels[s])
    plt.figure(); plt.imshow(predicted[s])

    # plt.figure(); plt.imshow(data[0, 0, 64, :, :])
    # plt.figure(); plt.imshow(data[0, 1, 64, :, :])
    # plt.figure(); plt.imshow(data[0, 2, 64, :, :])
    # plt.figure(); plt.imshow(data[0, 3, 64, :, :])
    plt.show()