import numpy as np
from nnunet.preprocessing.cropping import ImageCropper, crop_to_nonzero, crop_to_bbox, get_bbox_from_mask, \
    create_nonzero_mask, load_case_from_list_of_files


def get_bbox_from_mask_with_margin(mask, outside_value=0, margin=0.05):
    bbx = get_bbox_from_mask(mask, outside_value)
    sh = mask.shape
    for it, item in enumerate(bbx):
        marg = int((item[1] - item[0]) * margin)
        item[0] = int(max(item[0] - marg, 0))
        item[1] = int(min(item[1] + marg, sh[it]))
    return bbx


def crop_to_seg(data, seg=None, nonzero_label=-1, margin=0.05):
    """
    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :param margin: margin around image on each side (%)
    :return:
    """

    if seg is None:
        return data, seg, None


    mask = seg.copy()
    mask[mask == -1] = 0  # -1 is used to label unidentified voxels in nnunet
    nonzero_mask = create_nonzero_mask(mask)  # difference with crop_to_nonzero: mask instead of data
    bbox = get_bbox_from_mask_with_margin(nonzero_mask, 0, margin)  # difference with crop_to_nonzero

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


class ClassificationImageCropper(ImageCropper):
    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None, create_dummy_seg=False):
        if seg_file is None:
            assert create_dummy_seg
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
        if create_dummy_seg:
            seg = np.ones_like(data[:1])
        print("Shapes:")
        print(seg.shape)
        print(data.shape)
        return ClassificationImageCropper.crop(data, properties, seg)

    @staticmethod
    def crop(data, properties, seg=None):
        # always have an (empty) segmentation mask as input to the model. None in argument to match signature
        assert seg is not None
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop to nonzero:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        # Added:
        data, seg, bbox = crop_to_seg(data, seg, nonzero_label=-1)  # does not replace crop_to_nonzero
        shape_after_crop_to_seg = data.shape
        print("before crop to roi:", shape_after, "after crop to roi:", shape_after_crop_to_seg, "spacing:",
              np.array(properties["original_spacing"]), "\n", flush=True)

        properties["crop_bbox"] = bbox
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape

        return data, seg, properties
