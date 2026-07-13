# Note

## Usage

```bash
cd segformer3d/SegFormer3D/experiments/brats_2017/template_experiment

conda activate seg3d

accelerate launch --config_file ./gpu_accelerate.yaml ./run_experiment.py
python test.py
```

## Memory Usage Control

torch.Size([batch_size, 4, 96, 96, 96])
augmentations/augmentations.py > RandSpatialCropSamplesd > `num_samples`
config.yaml > train_dataloader_args > `batch_size`
batch_size = num_samples * batch_size

## Preprocess AI suggestions

### 1. Summary of the Entire File

The brats2017_seg_preprocess.py script is a data-preparation pipeline that reads raw 3D MRI scans in NIfTI format (.nii.gz), processes them, and outputs preprocessed tensor files (.pt) saved to disk in parallel using multiprocessing.

It includes:

- ConvertToMultiChannelBasedOnBrats2017Classes: The label grouping class we discussed.
- Brats2017Task1Preprocess: The core dataset class containing normalization, re-orientation, cropping, and saving logic.
- Visualization Helpers (animate & viz): Matplotlib functions to view slices of the 3D volume as an animation.

### 2. How the Data is Altered (Transforms Applied)

Besides the label multichannel conversion, the MRI scans and labels undergo several key transformations:

- Intensity Normalization (Images Only): MRI voxel intensities are scaled to the [0, 1] range using scikit-learn's MinMaxScaler on a per-volume basis.
- Orientation Standardization: The spatial coordinate space is aligned to RAS (Right, Anterior, Superior) using monai.transforms.Orientation.
- Spatial Cropping: The spatial volumes are cropped from the original size (240, 240, 155) to a standard grid size of (128, 128, 128) via the crop_brats2021_zero_pixels method.
- Transverse Axis Alignment: Axes 1 and 3 are swapped (.swapaxes(1, 3)) to align the data along the transverse/axial plane.
- Multi-Modal Stacking: The 4 separate modality scans (Flair, T1w, T1gd, T2w) are stacked along the channel dimension, resulting in a single multi-channel tensor of shape (4, 128, 128, 128).

### 3. BraTS-Specific vs. Generalizable Components

When adapting this script for a new dataset, categorize the logic as follows:

#### ❌ BraTS-Specific (Needs replacement or modification)

- File Structure & Modality Mapping (self.MRI_CODE): BraTS names files using suffixes like_0000.nii.gz through _0003.nii.gz to represent Flair, T1w, T1gd, and T2w modalities. You must update self.get_modality_fp and __getitem__ to match your new dataset's naming conventions and channel count.
- Hardcoded Cropping (crop_brats2021_zero_pixels): The slice indices [56:184, 56:184, 13:141] are tailored specifically for BraTS's default (240, 240, 155) grid size. If your new dataset has a different spatial resolution, this crop will either throw an out-of-bounds error or crop the wrong region. You should replace this with a dynamic cropping transform (e.g., MONAI's CropForeground or SpatialPad/Resize).
- Sub-region Grouping (ConvertToMultiChannelBasedOnBrats2017Classes): The target classes (Whole Tumor, Tumor Core, Enhancing Tumor) are unique to BraTS evaluation. You will need to define your own mapping for the new dataset's target classes.

#### Generalizable (Keep and reuse)

- NIfTI Loading (load_nifti): Reading volumetric files using nibabel and retrieving their affine matrices is standard for most 3D medical datasets.
- Orientation Standardization (orient): Standardizing scans to a consistent orientation (like RAS) is crucial for 3D CNNs to learn spatial features effectively.
- Intensity Normalization (normalize): Scaling voxels to [0, 1] or zero-mean/unit-variance is universally recommended for training neural networks.
- Parallel Save Skeleton (__call__ / process): The multiprocessing loop that saves output tensors to .pt files is highly reusable and avoids pipeline bottlenecks.

# wand API

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
