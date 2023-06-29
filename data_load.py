"""
adapted from https://github.com/Shifts-Project/shifts/tree/main/mswml
"""
import numpy as np
import os
from glob import glob
import re
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, RandCropByPosNegLabeld,
    Spacingd, ToTensord, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd,
    RandScaleIntensityd, RandSpatialCropSamplesd,ConcatItemsd)
from scipy import ndimage


def     get_train_transforms(I=['FLAIR']):
    """ Get transforms for training on FLAIR images and ground truth:
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Normalises intensity
    - Applies augmentations
    - Crops out 32 patches of shape [96, 96, 96] that contain lesions
    - Converts to torch.Tensor()
    """

    transform = Compose(
        [
            LoadImaged(keys=I+["label"]),
            #AddChanneld(keys=I+["label"]),
            NormalizeIntensityd(keys=I, nonzero=True),
            RandShiftIntensityd(keys=I, offsets=0.1, prob=1.0),
            RandScaleIntensityd(keys=I, factors=0.1, prob=1.0),
            RandCropByPosNegLabeld(keys=I+["label"],
                                   label_key="label", image_key=I[0],
                                   spatial_size=(128, 128, 128), num_samples=32,
                                   pos=4, neg=1),
            RandSpatialCropd(keys=I+["label"],
                             roi_size=(96, 96, 96),
                             random_center=True, random_size=False),
            RandFlipd(keys=I+["label"], prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=I+["label"], prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=I+["label"], prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=I+["label"], prob=0.5, spatial_axes=(0, 2)),
            RandAffined(keys=I+["label"], mode=('bilinear', 'nearest'),
                        prob=1.0, spatial_size=(96, 96, 96),
                        rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                        scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
            ToTensord(keys=I+["label"]),
            ConcatItemsd(keys=I, name="image", dim=0)
        ]
    )
    # transform.set_random_state(seed=seed)

    return transform


def get_ssl_transforms(I=['FLAIR']):
    """ Get transforms for training on FLAIR images :
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Normalises intensity
    - Applies augmentations
    - Crops out 32 patches of shape [96, 96, 96] that contain lesions
    - Converts to torch.Tensor()
    """
    transform = Compose(
        [
            LoadImaged(keys=I),
            AddChanneld(keys=I),
            NormalizeIntensityd(keys=I, nonzero=True),
            RandShiftIntensityd(keys=I, offsets=0.1, prob=1.0),
            RandScaleIntensityd(keys=I, factors=0.1, prob=1.0),
            RandSpatialCropSamplesd(keys=I,
                                    num_samples=32,
                                    roi_size=(96, 96, 96),
                                    random_center=True, random_size=False),
            RandFlipd(keys=I, prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=I, prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=I, prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=I, prob=0.5, spatial_axes=(0, 2)),
            RandAffined(keys=I, mode=('bilinear'),
                        prob=1.0, spatial_size=(96, 96, 96),
                        rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                        scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
            ToTensord(keys=I),
            ConcatItemsd(keys=I, name="image", dim=0)
        ]
    )
    # transform.set_random_state(seed=seed)

    return transform


def get_val_transforms(I=['FLAIR']): #keys=["image", "label"], image_keys=["image"]):
    """ Get transforms for testing on FLAIR images and ground truth:
    - Loads 3D images and masks from Nifti file
    - Adds channel dimention
    - Applies intensity normalisation to scans
    - Converts to torch.Tensor()
    """
    return Compose(
        [
            LoadImaged(keys=I+["label"]),
            AddChanneld(keys=I+["label"]),
            NormalizeIntensityd(keys=I, nonzero=True),
            ToTensord(keys=I+["label"]),
        ]
    )


def get_train_dataloader(scan_paths, gts_paths, num_workers, cache_rate=0.1, seed=1, I=['FLAIR']):
    """
    Get dataloader for training
    Args:
      scan_paths: `list`, list of paths to directories with different modality images.
      gts_path:  `str`, path to directory with ground truth lesion segmentation
                    binary masks images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      modalities: `list`, list of modalities to include in the data loader.
    Returns:
      monai.data.DataLoader() class object.
    """
    # Collect all modality images sorted
    all_modality_images = {}
    assert isinstance(scan_paths, list), "scan_paths must be a list"
    for modality in I:
        modality_images = []
        for scan_path in scan_paths:
            modality_path = os.path.join(scan_path, modality.lower())
            modality_images += sorted(glob(os.path.join(modality_path, "*.nii.gz")), key=lambda i: int(re.sub('\D', '', i)))
        all_modality_images[modality] = modality_images

    # Check all modalities have same length
    assert all(len(x) == len(all_modality_images[I[0]]) for x in all_modality_images.values()), "All modalities must have the same number of images"

    # Collect all corresponding ground truths
    segs = []
    if isinstance(gts_paths, list):
        for path in gts_paths:
            segs += sorted(glob(os.path.join(path, "*.nii.gz")),  key=lambda i: int(re.sub('\D', '', i)))
    elif gts_paths is not None:
        segs = sorted(glob(os.path.join(gts_paths, "*.nii.gz")),  key=lambda i: int(re.sub('\D', '', i)))

    assert len(all_modality_images[I[0]]) == len(segs), "Number of multi-modal images and ground truths must be the same"

    files = []
    for i in range(len(segs)):
        file_dict = {"label": segs[i]}
        for modality in I:
            file_dict[modality] = all_modality_images[modality][i]
        files.append(file_dict)

    print("Number of training files:", len(files))
    print(I)
    train_transforms = get_train_transforms(I)
    ds = CacheDataset(data=files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=True,  num_workers=num_workers)



def get_val_dataloader(scan_paths, gts_paths, num_workers, cache_rate=0.1, bm_paths=None, I=['FLAIR']):
    """
    Get dataloader for validation and testing. Either with or without brain masks.

    Args:
      scan_paths: `list`, list of paths to directories with different modality images.
      gts_path:  `str`, path to directory with ground truth lesion segmentation
                    binary masks images.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks.
      I: `list`, list of I to include in the data loader.
    Returns:
      monai.data.DataLoader() class object.
    """
    # Collect all modality images sorted
    all_modality_images = {}
    assert isinstance(scan_paths, list), "scan_paths must be a list"
    for modality in I:
        modality_images = []
        for scan_path in scan_paths:
            modality_path = os.path.join(scan_path, modality.lower())
            modality_images += sorted(glob(os.path.join(modality_path, "*.nii.gz")), key=lambda i: int(re.sub('\D', '', i)))
        all_modality_images[modality] = modality_images

    # Check all modalities have same length
    assert all(len(x) == len(all_modality_images[I[0]]) for x in all_modality_images.values()), "All I must have the same number of images"

    segs = []
    if isinstance(gts_paths, list):
        for path in gts_paths:
            segs += sorted(glob(os.path.join(path, "*gt*.nii.gz")), key=lambda i: int(re.sub('\D', '', i)))
    else:
        segs = sorted(glob(os.path.join(gts_paths, "*.nii.gz")), key=lambda i: int(re.sub('\D', '', i)))

    if bm_paths is not None:
        bms = []
        if isinstance(bm_paths, list):
            for path in bm_paths:
                bms += sorted(glob(os.path.join(path, "*.nii.gz")), key=lambda i: int(re.sub('\D', '', i)))
        else:
            bms = sorted(glob(os.path.join(bm_paths, "*.nii.gz")), key=lambda i: int(re.sub('\D', '', i)))

        assert len(all_modality_images[I[0]]) == len(segs) == len(bms), f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms)]}"

        files = []
        for i in range(len(segs)):
            file_dict = {"label": segs[i], "brain_mask": bms[i]}
            for modality in I:
                file_dict[modality] = all_modality_images[modality][i]
            files.append(file_dict)

        val_transforms = get_val_transforms(keys=I + ["label", "brain_mask"])
    else:
        assert len(all_modality_images[I[0]]) == len(segs), f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs)]}"

        files = []
        for i in range(len(segs)):
            file_dict = {"label": segs[i]}
            for modality in I:
                file_dict[modality] = all_modality_images[modality][i]
            files.append(file_dict)

        val_transforms = get_val_transforms(I)

    print("Number of validation files:", len(files))

    ds = CacheDataset(data=files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)





def get_flair_dataloader(flair_path, num_workers, cache_rate=0.1, batch_size=1, bm_path=None, ssl=False, seed=None):
    """
    Get dataloader with FLAIR images only for inference

    Args:
      flair_path: `str`, path to directory with FLAIR images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks.
    Returns:
      monai.data.DataLoader() class object.
    """

    if ssl:
        flair = sorted(glob(os.path.join(flair_path, "*_isovox.nii.gz")),
                       key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted

    else:
        flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                       key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted

    if bm_path is not None:
        bms = sorted(glob(os.path.join(bm_path, "*isovox_fg_mask.nii.gz")),
                     key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding brain masks

        assert len(flair) == len(bms), f"Some files must be missing: {[len(flair), len(bms)]}"

        files = [{"image": fl, "brain_mask": bm} for fl, bm in zip(flair, bms)]

        val_transforms = get_val_transforms(keys=["image", "brain_mask"])
    else:
        files = [{"image": fl} for fl in flair]
        if ssl:
            transforms = get_ssl_transforms()
            if seed:
                transforms.set_random_state(seed)
        else:
            transforms = get_val_transforms(keys=["image"])

    print("Number of FLAIR files:", len(files))

    ds = CacheDataset(data=files, transform=transforms,
                      cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers)


def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
            current_voxels[:, 1],
            current_voxels[:, 2]] = 1
    return seg2
