import numpy as np


def central_pad_data_or_seg(np_image, target_size, outside_val=0):
    """
    Centrally crops or pads a 4D numpy array (channels, x, y, z) to the target size.

    Args:
        np_image (numpy.ndarray): Image or segmentation data of shape (c, x, y, z).
        target_size (list or tuple): Desired size [x, y, z].
        outside_val (int or float, optional): Value to pad with. Defaults to 0.

    Returns:
        numpy.ndarray: Padded or cropped image or segmentation data of shape (c, target_x, target_y, target_z).
    """
    target_size = np.asarray([np_image.shape[0]] + list(target_size), dtype=int)
    assert len(np_image.shape) == 4, "data must be (c, x, y, z)"

    # Step 1: Pre-cropping to match target_size for dimensions that are too large
    cropping_slices = []
    for dim, (current_size, target_dim) in enumerate(zip(np_image.shape, target_size)):
        if dim == 0:  # Channel dimension, no cropping or padding
            cropping_slices.append(slice(None))
        elif current_size > target_dim:
            # Center-crop the dimension to fit target_size
            start = (current_size - target_dim) // 2
            end = start + target_dim
            cropping_slices.append(slice(start, end))
        else:
            # No cropping needed
            cropping_slices.append(slice(None))

    # Apply cropping
    np_image = np_image[tuple(cropping_slices)]

    # Step 2: Initialize output image with padding value for final padding
    output_image = np.full(target_size, outside_val, dtype=np_image.dtype)

    # Step 3: Padding
    offsets = []
    for dim, (current_size, target_dim) in enumerate(zip(np_image.shape, target_size)):
        if dim == 0:  # Channel dimension
            offsets.append(slice(None))
        else:
            # Calculate padding offset
            pad_before = max((target_dim - current_size) // 2, 0)
            offsets.append(slice(pad_before, pad_before + current_size))

    # Fill the output image with the cropped and padded data
    output_image[tuple(offsets)] = np_image

    return output_image


def central_pad(data, target_size, properties, seg):
    """
    Applies central padding to both data and segmentation arrays.

    Args:
        data (numpy.ndarray): Image data of shape (c, x, y, z).
        target_size (list or tuple): Desired size [x, y, z].
        properties (dict): Metadata dictionary to update with padding information.
        seg (numpy.ndarray): Segmentation data of shape (c, x, y, z).

    Returns:
        tuple: Padded data, padded segmentation, and updated properties.
    """
    assert not ((data is None) and (seg is None)), "Both data and seg cannot be None."
    if data is not None:
        assert len(data.shape) == 4, "Data must be a 4D array (c, x, y, z)."
    if seg is not None:
        assert len(seg.shape) == 4, "Segmentation must be a 4D array (c, x, y, z)."

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)

    print(f"Applying uniform padding from {shape} to {target_size}...")
    if data is not None:
        data = central_pad_data_or_seg(data, target_size, outside_val=0)
    if seg is not None:
        seg = central_pad_data_or_seg(seg, target_size, outside_val=0)

    properties['size_after_central_pad'] = target_size
    return data, seg, properties
