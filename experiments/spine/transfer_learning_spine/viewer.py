import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


class SliceViewer:
    def __init__(self, volume, fig, ax, **kwargs):
        self.volume = volume
        # Start at the middle slice of the 3D volume
        self.slices = volume.shape[0]
        self.ind = self.slices // 2

        # Set up the plot
        self.fig = fig
        self.ax = ax
        self.im = self.ax.imshow(self.volume[self.ind, :, :], **kwargs)
        self.update_title()

        # Connect the scroll event to our function
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def update_title(self):
        self.ax.set_title(f'Use mouse scroll to view slices\nSlice {self.ind}/{self.slices}')

    def on_scroll(self, event):
        # event.button can be 'up' or 'down' depending on scroll direction
        if event.button == 'down':
            self.ind = (self.ind + 1) if self.ind < self.slices-1 else self.slices-1
        elif event.button == 'up':
            self.ind = (self.ind - 1) if self.ind > 0 else 0

        # Update the image data and redraw
        self.im.set_data(self.volume[self.ind, :, :])
        self.update_title()
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    case_id = f'case_{num:04d}'
    output_dir = Path('C:/Users/Ryan Deng/SegFormer3D/data/spine')
    image_path = output_dir / 'preprocessed' / case_id / f'{case_id}_image.pt'
    label_path = output_dir / 'preprocessed' / case_id / f'{case_id}_label.pt'
    predicted_path = 'predicted.pt' if os.path.exists('predicted.pt') else label_path

    # load data
    image = torch.load(image_path).cpu().numpy()
    image = image[0, ...]
    image = np.flip(image.transpose(2, 0, 1), axis=(1, 2))

    label = torch.load(label_path).cpu().numpy()
    label = label[0, ...]
    label = np.flip(label.transpose(2, 0, 1), axis=(1, 2))

    predicted = torch.load(predicted_path).cpu().numpy()
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
