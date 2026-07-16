import os
import random
from pathlib import Path

import numpy as np
import torch
import nibabel
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='monai.utils.module')
from monai.data import MetaTensor
from monai.transforms import Orientation, EnsureType
from tqdm.contrib.concurrent import process_map


class SpinePreprocess:
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        save_dir: Path,
        num_classes: int,
        vmin: float,
        vmax: float,
        force_reprocess: bool = False,
        max_workers: int = 1
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.vmin = vmin
        self.vrange = vmax - vmin
        self.force_reprocess = force_reprocess
        self.max_workers = max_workers
        fnames = list(self.image_dir.glob('*.nii'))
        self.fnames = [v.name for v in fnames]


    def preprocess(self, raw, is_label: bool) -> np.ndarray:
        data = raw.get_fdata()
        affine = raw.affine

        # convert to tensor
        data = torch.from_numpy(data)

        if is_label:
            # (X, Y, Z) -> (1, X, Y, Z)
            # don't convert to one-hot encoding
            data = data.unsqueeze(0).to(torch.int8)

        else:
            # (X, Y, Z) -> (1, X, Y, Z)
            data = data.unsqueeze(0).to(torch.float32)
            # intensity normalization
            data = (data - self.vmin) / self.vrange
            data = torch.clamp(data, 0, 1)

        # orientation standardization
        # (C, X, Y, Z) -> (C, Right, Anterior, Superior) -> (C, Superior, Anterior, Right)
        data = MetaTensor(x=data, affine=affine)
        data = Orientation(axcodes='RAS')(data)
        data = data.swapdims(1, 3)

        # remove metadata
        data = EnsureType(data_type='tensor', track_meta=False)(data)
        return data


    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image_path = self.image_dir / fname
        label_path = self.label_dir / fname
        assert image_path.exists()
        assert label_path.exists()

        image_raw = nibabel.load(image_path)
        label_raw = nibabel.load(label_path)

        image = self.preprocess(image_raw, is_label=False)
        label = self.preprocess(label_raw, is_label=True)
        return image, label


    def __len__(self):
        return len(self.fnames)


    def __call__(self):
        print('started preprocessing Spine Dataset...')
        process_map(self.process, range(self.__len__()), max_workers=self.max_workers, chunksize=16)
        print('finished preprocessing Spine Dataset...')


    def process(self, idx):
        fname = self.fnames[idx]
        case_id = fname.split('.', 1)[0]
        case_dir = self.save_dir / case_id
        image_path = case_dir / f'{case_id}_image.pt'
        label_path = case_dir / f'{case_id}_label.pt'
        if all([
            os.path.exists(image_path),
            os.path.exists(label_path),
            self.force_reprocess == False
        ]):
            return

        os.makedirs(case_dir, exist_ok=True)
        image, label = self.__getitem__(idx)
        torch.save(image, image_path)
        torch.save(label, label_path)


if __name__ == '__main__':
    raw_data = Path('C:/Users/Ryan Deng/vista_robotics_spine_2d_3d_registration/dataset/segmentation')
    raw_data = raw_data / 'spine_segmentation_nnunet_v2'
    # raw_data = raw_data / 'test'
    output_dir = Path('C:/Users/Ryan Deng/SegFormer3D/data/spine')
    spine_prep = SpinePreprocess(
        image_dir = raw_data / 'volumes',
        label_dir = raw_data / 'segmentations',
        save_dir = output_dir / 'preprocessed',
        num_classes = 26,
        vmin = -1e4,
        vmax = 1e4,
        force_reprocess = False,
        max_workers = 2  # disk io bottleneck
    )
    spine_prep()

    # train validation test split
    splits = {
        'train': 0.987,
        'validation': 0.003,
        'test': 0.01
    }

    all_set = list((output_dir / 'preprocessed').glob('*'))
    random.seed(len(all_set))
    random.shuffle(all_set)

    split_indices = np.cumsum([int(len(all_set) * v) for v in splits.values()])[:-1]
    indices = np.split(np.arange(len(all_set)), split_indices)
    for name, idx in zip(splits.keys(), indices):
        with open(output_dir / f'{name}.csv', 'w') as f:
            f.write('data_path,case_name')
            for i in idx:
                path = all_set[i]
                rpath = '../../../' + path.relative_to(output_dir.parents[1]).as_posix()
                f.write(f'\n{rpath},{path.stem}')
