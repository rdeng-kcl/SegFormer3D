import sys
import argparse
from pathlib import Path
from torchinfo import summary

sys.path.append("../../../")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module") # ignore monai warnings

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

    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num', required=False, type=int, help='number of the test case id, default all cases in the split')
    parser.add_argument('-p', '--plot', action='store_true', help='plot the results')
    parser.add_argument('-s', '--save', action='store_true', help='save the results as ./{case_id}_pred.pt')
    parser.add_argument('-c', '--checkpoint', type=str, default='./model_checkpoints/dicece_success/pytorch_model.bin', help='path to the model checkpoint')
    parser.add_argument('--dir', type=str, default='../../../data/spine/preprocessed', help='path to the preprocessed data directory')
    parser.add_argument('--config', type=str, default='./config.yaml', help='path to the config file')
    parser.add_argument('--split', type=str, default='../../../data/spine/test.csv', help='path to the data split file for testing all cases')
    parser.add_argument('--summary', action='store_true', help='print the model summary')
    args = parser.parse_args()

    cuda = "cuda:0"
    config = load_config(args.config)
    if args.num is None:
        with open(args.split, 'r') as f:
            case_nums = [int(line.strip().split(',')[1].replace('case_', '')) for line in f.readlines()[1:]]
    else:
        case_nums = [args.num]

    # load model
    model = build_architecture(config)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(cuda)
    if args.summary:
        tensor_shape = (
            config["sliding_window_inference"]["sw_batch_size"],
            1,
            config["sliding_window_inference"]["roi"][0],
            config["sliding_window_inference"]["roi"][1],
            config["sliding_window_inference"]["roi"][2]
        )
        summary(model, input_size=tensor_shape)

    # test all cases
    dice_list = []
    for case_num in case_nums:

        # load data
        case_id = f'case_{case_num:04d}'
        output_dir = Path(args.dir)
        image_path = output_dir / case_id / f'{case_id}_image.pt'
        label_path = output_dir / case_id / f'{case_id}_label.pt'
        image = torch.load(image_path).cpu()
        label = torch.load(label_path).cpu()

        # sliding window inference and calculate metrics
        sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"],
        )

        logits = sliding_window_inference.forward(image.to(cuda).unsqueeze(0), model)
        predicted = sliding_window_inference.to_label(logits.squeeze(0)).cpu().as_tensor().to(torch.int8)

        if args.save:
            fname = f"{case_id}_pred.pt"
            torch.save(predicted, fname)
            print(f"Saved {fname}")

        _, dice = sliding_window_inference.calc_loss_dice_metric(logits, label.unsqueeze(0), criterion=None, device=cuda)
        print(f"{case_id} dice: {dice:.2f} %")
        dice_list.append(dice)
        torch.cuda.empty_cache()

        if args.plot:
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

    if args.num is None:
        print(f"Average dice: {sum(dice_list)/len(dice_list):.2f} %")
