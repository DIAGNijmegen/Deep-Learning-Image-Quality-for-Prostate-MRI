from universalclassifier.training.data_augmentation.transforms import RescaleSegmentationTransform
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, NumpyToTensor

def get_moreDA_augmentation(dataloader_train, dataloader_val, patch_size, params, pin_memory=True):
    # Heavy augmentation for training
    train_transforms = [
        # Spatial Transform with rotation and scaling (random crop included)
        SpatialTransform(
            patch_size,
            do_rotation=False,
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=False,
            scale=params.get("scale_range"),
            border_mode_data="constant",
            border_cval_data=0,
            border_mode_seg="constant",
            order_seg=1,
            random_crop=False
        ),
        #MirrorTransform(params.get("mirror_axes")),

        # Intensity adjustments (MRI-specific)
        #GammaTransform(params.get("gamma_range"), invert_image=True, retain_stats=True, p_per_sample=0.05),
        #ContrastAugmentationTransform(p_per_sample=0.10),

        # Additional transforms that are required for both training and validation
        RescaleSegmentationTransform(params.get("num_seg_classes")),
        RemoveLabelTransform(-1, 0),
        NumpyToTensor(['data'], 'float'),
        NumpyToTensor(['target'], 'long')
    ]
    train_transforms = Compose(train_transforms)

    # Minimal augmentation for validation.
    # Usually for validation, we want only the necessary preprocessing steps to bring the data
    # into the correct format, without any random or heavy augmentation.
    val_transforms = [
        # Do any necessary rescaling or segmentation transform that is not random.
        RescaleSegmentationTransform(params.get("num_seg_classes")),
        RemoveLabelTransform(-1, 0),
        NumpyToTensor(['data'], 'float'),
        NumpyToTensor(['target'], 'long')
    ]
    val_transforms = Compose(val_transforms)

    # Create separate augmenters for training and validation
    batchgenerator_train = MultiThreadedAugmenter(
        dataloader_train,
        train_transforms,
        params.get('num_threads'),
        params.get("num_cached_per_thread"),
        pin_memory=pin_memory
    )
    batchgenerator_val = MultiThreadedAugmenter(
        dataloader_val,
        val_transforms,
        max(params.get('num_threads') // 2, 1),
        params.get("num_cached_per_thread"),
        pin_memory=pin_memory
    )

    return batchgenerator_train, batchgenerator_val