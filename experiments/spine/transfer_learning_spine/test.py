import sys
from pathlib import Path
from torchinfo import summary

sys.path.append("../../../")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module") # ignore monai warnings (not working)

import torch
import numpy as np
from architectures.build_architecture import build_architecture
from dataloaders.build_dataset import build_dataset, build_dataloader
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.append("../../brats_2017/template_experiment")
from run_experiment import load_config
from metrics.segmentation_metrics import SlidingWindowInference
from viewer import SliceViewer


if __name__ == '__main__':

    num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    case_id = f'case_{num:04d}'
    output_dir = Path('../../../data/spine')
    image_path = output_dir / 'preprocessed' / case_id / f'{case_id}_image.pt'
    label_path = output_dir / 'preprocessed' / case_id / f'{case_id}_label.pt'

    # load data
    cuda = "cuda:0"
    image = torch.load(image_path).cpu()
    label = torch.load(label_path).cpu()

    config = load_config("config.yaml")

    # load model
    model = build_architecture(config)
    checkpoint_file = "model_checkpoints/background_yes_failed/pytorch_model.bin"
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    sliding_window_inference = SlidingWindowInference(
        config["sliding_window_inference"]["roi"],
        config["sliding_window_inference"]["sw_batch_size"],
    )

    logits = sliding_window_inference.forward(image.to(cuda).unsqueeze(0), model.to(cuda)).squeeze(0)
    predicted = sliding_window_inference.post_transform(logits).cpu()

    torch.save(predicted, "predicted.pt")

    image = image.numpy()
    image = image[0, ...]
    image = np.flip(image.transpose(2, 0, 1), axis=(1, 2))
    label = label.numpy()
    label = label[0, ...]
    label = np.flip(label.transpose(2, 0, 1), axis=(1, 2))
    predicted = predicted.numpy()
    predicted = predicted[0, ...]
    predicted = np.flip(predicted.transpose(2, 0, 1), axis=(1, 2))

    c10 = plt.cm.tab10(np.linspace(0, 1, 10))
    c26 = np.vstack((np.array([[0,0,0,1]]), c10, c10, c10[:5, :]))
    cmap_custom = ListedColormap(c26)
    norm_custom = BoundaryNorm(np.arange(0, 27) - 0.5, cmap_custom.N)

    # show image and label
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))
    vi = SliceViewer(image, fig, ax1)
    vl = SliceViewer(label, fig, ax2, cmap=cmap_custom, norm=norm_custom, interpolation='nearest')
    vp = SliceViewer(predicted, fig, ax3, cmap=cmap_custom, norm=norm_custom, interpolation='nearest')
    plt.show()