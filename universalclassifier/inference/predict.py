import os
import shutil
import nibabel as nib
import numpy as np
import torch
import cv2
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *
from universalclassifier.training.model_restore import load_model_and_checkpoint_files
from universalclassifier.inference.export import save_output
from typing import Union, Tuple, List
from copy import deepcopy
import SimpleITK as sitk
# Install scipy if not already installed
import subprocess
import sys
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

try:
    import scipy
except ImportError:
    print("scipy not found. Installing scipy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    import scipy  # Import again after installation

from scipy.ndimage import zoom


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.feature_maps = None
        self.gradients = None
        self.hook = self.model.mixed_3b.register_forward_hook(self.save_feature_maps)
        self.hook_backward = self.model.mixed_3b.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        output = self.model(x)
        return output
    

def burn_text_into_volume(volume, text, z_indices, position,
                          font_scale=1.0, thickness=2, value=1.0):
    """
    Draws text onto selected axial slices of a 3D volume.

    volume: np.ndarray (Z, Y, X)
    text: str, the message to burn in
    z_indices: list of Z slice indices
    position: (x, y) tuple of pixel location within the slice
    font_scale: float, size of the text
    thickness: int, stroke width
    value: float, voxel value to write (e.g., 1.0)
    """
    print("z_indices =", z_indices)
    for z in z_indices:
        if 0 <= z < volume.shape[0]:
            slice_img = volume[z]
            canvas = np.zeros_like(slice_img, dtype=np.uint8)

            cv2.putText(
                canvas, text, position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=int(value * 255),
                thickness=thickness,
                lineType=cv2.LINE_AA
            )
            # Burn into the volume
            volume[z][canvas > 0] = value



def plot_or_save_slices(image, path=None):
    """
    Plots or saves a grid of slices from a 3D or 4D image.

    Parameters:
    - image: 3D or 4D numpy array representing the image.
    - path: Optional. If provided, the grid will be saved to this path instead of being plotted.
    """
    if image.ndim not in [3, 4]:
        raise ValueError("Input image must be a 3D or 4D array")

    if image.ndim == 4 and image.shape[3] != 3:
        raise ValueError("For 4D input, the last dimension must be 3 (RGB channels)")

    # Identify the dimension with the smallest length
    min_dim = np.argmin(image.shape[:3])

    if min_dim == 0:
        num_slices = image.shape[0]
        if image.ndim == 3:
            slice_func = lambda img, idx: img[idx, :, :]
        else:
            slice_func = lambda img, idx: img[idx, :, :, :]
    elif min_dim == 1:
        num_slices = image.shape[1]
        if image.ndim == 3:
            slice_func = lambda img, idx: img[:, idx, :]
        else:
            slice_func = lambda img, idx: img[:, idx, :, :]
    else:
        num_slices = image.shape[2]
        if image.ndim == 3:
            slice_func = lambda img, idx: img[:, :, idx]
        else:
            slice_func = lambda img, idx: img[:, :, idx, :]

    num_cols = int(np.ceil(np.sqrt(num_slices)))
    num_rows = int(np.ceil(num_slices / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(num_rows):
        for j in range(num_cols):
            slice_index = i * num_cols + j
            if slice_index < num_slices:
                axes[i, j].imshow(slice_func(image, slice_index), cmap='gray' if image.ndim == 3 else None)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def generate_grad_cam(model, input_tensor, target_class):
    model.eval()
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor.requires_grad_(True)

    # Move input tensor to the same device as the model
    input_tensor = input_tensor.to(next(model.parameters()).device)

    output = model(input_tensor)

    # Ensure output is a tensor, not a list
    if isinstance(output, list):
        output = output[0]  # Assuming the first element is the relevant output

    target = output[0, target_class]
    target.backward()

    gradients = model.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])
    feature_maps = model.feature_maps.detach()

    for i in range(feature_maps.size(1)):
        feature_maps[:, i, :, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    heatmap = torch.maximum(heatmap, torch.tensor(0.0).to(heatmap.device))
    heatmap /= torch.max(heatmap)

    return heatmap.cpu().numpy()


def overlay_heatmap(image, heatmap):
    # Calculate the zoom factors for each dimension
    zoom_factors = (
        image.shape[0] / heatmap.shape[0],
        image.shape[1] / heatmap.shape[1],
        image.shape[2] / heatmap.shape[2]
    )

    # Resize heatmap using 3D interpolation
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)

    # Ensure the resized heatmap has the same shape as the image
    assert heatmap_resized.shape == image.shape, "Resized heatmap shape does not match image shape"

    # Initialize the colored heatmap array
    heatmap_colored = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3), dtype=np.uint8)

    # Convert each slice of the heatmap to uint8 and apply color map
    for z in range(image.shape[2]):
        heatmap_slice = np.uint8(255 * (1 - heatmap_resized[:, :, z]))
        heatmap_colored[:, :, z, :] = cv2.applyColorMap(heatmap_slice, cv2.COLORMAP_JET)

    # Convert image to uint8 if it's not already
    if image.dtype != np.uint8:
        image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))

    opacity = heatmap_resized / heatmap_resized.max()

    # Superimpose the heatmap on the image
    superimposed_img = np.round(heatmap_colored * opacity[..., np.newaxis] * 0.7 + image[..., np.newaxis]).astype(
        np.uint8)

    superimposed_img = np.clip(superimposed_img, 0, 255)

    return superimposed_img


def convert_mha_to_nii_gz(mha_path, output_path):
    image = sitk.ReadImage(str(mha_path))
    sitk.WriteImage(image, str(output_path))

def save_heatmap_as_mha(heatmap,
                        output_path,
                        preprocessed_shape,
                        original_image_sitk: sitk.Image,
                        draw_box: bool = True,
                        burn_text: bool = True):
    """
    Save a heatmap as .mha, resized to preprocessed shape, padded to original shape,
    and embedded with bounding box and/or burned-in text.

    Parameters:
    - heatmap: np.ndarray (3D, e.g., 7x38x38)
    - output_path: str, path to save .mha
    - preprocessed_shape: tuple (e.g., 15x300x300)
    - original_image_sitk: sitk.Image, the original image used for copying spacing, origin, direction
    - draw_box: bool, whether to draw bounding box around model region
    - burn_text: bool, whether to embed "MODEL REGION" text
    """
    # Step 1: Resize to preprocessed shape
    zoom_factors = (
        preprocessed_shape[0] / heatmap.shape[0],
        preprocessed_shape[1] / heatmap.shape[1],
        preprocessed_shape[2] / heatmap.shape[2],
    )
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)

    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / heatmap_resized.max()

    heatmap_resized = heatmap_resized.astype(np.float32)

    # Step 2: Pad to original shape
    original_shape = original_image_sitk.GetSize()[::-1]  # Convert to (Z, Y, X)
    pad_z = (original_shape[0] - preprocessed_shape[0]) // 2
    pad_y = (original_shape[1] - preprocessed_shape[1]) // 2
    pad_x = (original_shape[2] - preprocessed_shape[2]) // 2

    pad_z_extra = (original_shape[0] - preprocessed_shape[0]) % 2
    pad_y_extra = (original_shape[1] - preprocessed_shape[1]) % 2
    pad_x_extra = (original_shape[2] - preprocessed_shape[2]) % 2

    pad_width = (
        (pad_z, pad_z + pad_z_extra),
        (pad_y, pad_y + pad_y_extra),
        (pad_x, pad_x + pad_x_extra)
    )

    heatmap_padded = np.pad(
        heatmap_resized,
        pad_width,
        mode='constant',
        constant_values=0
    ).astype(np.float32)

    if draw_box:
        z0, z1 = pad_width[0][0], pad_width[0][0] + preprocessed_shape[0]
        y0, y1 = pad_width[1][0], pad_width[1][0] + preprocessed_shape[1]
        x0, x1 = pad_width[2][0], pad_width[2][0] + preprocessed_shape[2]

        # Draw full rectangle on every axial slice (z direction)
        for z in range(z0, z1):
            heatmap_padded[z, y0, x0:x1] = 1.0        # top edge
            heatmap_padded[z, y1 - 1, x0:x1] = 1.0    # bottom edge
            heatmap_padded[z, y0:y1, x0] = 1.0        # left edge
            heatmap_padded[z, y0:y1, x1 - 1] = 1.0    # right edge

    # Step 4: Burn in "MODEL REGION" text on central slices
    if burn_text:
        z_center = heatmap_padded.shape[0] // 2
        z_indices=list(range(z_center - 7, z_center + 8))
        burn_text_into_volume(
            heatmap_padded,
            text="MODEL REGION",
            z_indices=z_indices,
            position=(210, 160),
            font_scale=1.0,
            thickness=1,
            value=1.0
        )

    # Step 5: Save with metadata
    img = sitk.GetImageFromArray(heatmap_padded)
    img.SetSpacing(original_image_sitk.GetSpacing())
    img.SetOrigin(original_image_sitk.GetOrigin())
    img.SetDirection(original_image_sitk.GetDirection())

    sitk.WriteImage(img, output_path)


def process_input_folder(patient_folder: str, patient_id: str, study_id: str, input_folder: str, modality: str):
    """Processes a patient folder, converting the HBV file to .nii.gz format."""
    if modality == 'DWI':
        mha_path = Path(patient_folder) / f"{patient_id}_{study_id}_hbv.mha"
    if modality == 'T2W':
        mha_path = Path(patient_folder) / f"{patient_id}_{study_id}_t2w.mha"
    if mha_path.exists():
        hbv_nii_path = Path(input_folder) / f"{patient_id}_{study_id}_t2w_0000.nii.gz"
        convert_mha_to_nii_gz(mha_path, hbv_nii_path)
        return hbv_nii_path
    else:
        print(f"{modality} file not found for patient {patient_id} and study {study_id}")
        return None


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, f"input folder did not contain any images (expected to find .nii.gz file endings) in the {input_folder} folder"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def predict_from_folder(model: str, patient_folder_root: str, output_folder: str,
                        folds: Union[Tuple[int], List[int]], mixed_precision: bool = True,
                        overwrite_existing: bool = True, checkpoint_name: str = "model_final_checkpoint",
                        folders_format: bool = True, modality: str = ''):
    """Predicts from a folder of patient folders based on a subject list file."""
    maybe_mkdir_p(output_folder)
    print(f'Folders Format is set to {folders_format}')
    if folders_format:
        input_folder = Path("temp_nifti_inputs")
        maybe_mkdir_p(input_folder)
        subject_list_path = Path(patient_folder_root) / 'subject_list.txt'
        # Read the subject list file and process each patient/study
        with open(subject_list_path, 'r') as f:
            subject_list = f.read().splitlines()
        for subject in subject_list:
            patient_id, study_id, = subject.split('_') #CHANGE NUM
            patient_folder = Path(patient_folder_root) / (patient_id)

            if not patient_folder.is_dir():
                print(f"Patient folder not found for {patient_id}. Skipping.")
                continue

            hbv_nii_file = process_input_folder(patient_folder, patient_id, study_id, input_folder, modality)
            if not hbv_nii_file:
                continue  # Skip if HBV file was not found or converted
    else:
        input_folder = patient_folder_root  # Comment to infere in Folder of patient folders

    # After conversions, check input folder integrity and predict
    case_ids = check_input_folder_and_return_caseIDs(input_folder, 1)  # Assuming only HBV modality for simplicity

    output_files = [join(output_folder, f"{i}.npz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i.startswith(case_id)] for case_id in case_ids]

    seg_files = [None] * len(case_ids)  # Assuming no segmentation files are provided for inference

    # Run predictions
    predict_cases(model, list_of_lists, seg_files, output_files, folds, mixed_precision=mixed_precision,
                  overwrite_existing=overwrite_existing, checkpoint_name=checkpoint_name)

    # Clean up temporary input folder after predictions
    if folders_format:
        shutil.rmtree(Path("temp_nifti_inputs"), ignore_errors=True)


def predict_cases(model, list_of_lists_of_modality_filenames, seg_filenames, output_filenames, folds,
                  mixed_precision=True, overwrite_existing=False, checkpoint_name="model_final_checkpoint"):
    assert len(list_of_lists_of_modality_filenames) == len(output_filenames)
    assert len(list_of_lists_of_modality_filenames) == len(seg_filenames)

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".npz"):
            f, _ = os.path.splitext(f)
            f = f + ".npz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists_of_modality_filenames))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists_of_modality_filenames = [list_of_lists_of_modality_filenames[i] for i in not_done_idx]
        seg_filenames = [seg_filenames[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)

    for input_files, seg_file, output_filename in zip(list_of_lists_of_modality_filenames,
                                                      seg_filenames, output_filenames):

        print(f"=== Processing {input_files}, {seg_file}:")
        print("preprocessing...")
        d, s, properties = trainer.preprocess_patient(input_files, seg_file)
        d = trainer.combine_data_and_seg(d, s)

        print("predicting...")
        trainer.load_checkpoint_ram(params[0], False)
        pred = trainer.predict_preprocessed_data_return_pred_and_logits(d[None], mixed_precision=mixed_precision)[1]

        for p in params[1:]:
            trainer.load_checkpoint_ram(p, False)
            new_pred = \
                trainer.predict_preprocessed_data_return_pred_and_logits(d[None], mixed_precision=mixed_precision)[1]
            pred = [p + n_p for p, n_p in zip(pred, new_pred)]

        if len(params) > 1:
            pred = [p / len(params) for p in pred]

        print("exporting prediction...")
        from scipy.special import softmax

        # pred is a list of logits, typically with shape (2,)
        logits = np.array(pred)  # shape: (1, 2)
        probabilities = softmax(logits, axis=-1)  # shape: (1, 2)
        prediction = float(probabilities[..., 1])  # extract class 1 prob

        json_output_path = output_filename.replace(".npz", ".json")

        with open(json_output_path, "w") as f:
            json.dump({"high_quality_probability": prediction}, f, indent=2)

        print(f"Saved high-quality probability to {json_output_path}")

        # Generate and overlay Grad-CAM heatmap on preprocessed image
        model_wrapper = ModelWrapper(trainer.network).to(next(trainer.network.parameters()).device)

        target_class = np.argmax(pred[0])
        input_tensor = torch.tensor(d)

        heatmap = generate_grad_cam(model_wrapper, input_tensor, target_class)

        # Print shape for troubleshooting
        print(f"Preprocessed image shape (d): {d.shape}, Heatmap shape: {heatmap.shape}")

        # Ensure heatmap and preprocessed image (d) have the correct dimensions
        if heatmap.ndim != 3 or d.ndim != 4:
            raise ValueError("Heatmap must be 3D and preprocessed image must be 4D (with channel dimension)")

        # Remove channel dimension for visualization
        d = d[0]  # Assuming d has shape (1, depth, height, width)

        input_path = list_of_lists_of_modality_filenames[0][0]
        original_img_sitk = sitk.ReadImage(input_path)

        heatmap_image = overlay_heatmap(d, heatmap)

        #overlay_output_path = output_filename.replace('.npz', f'_Overlay.jpg')
        #heat_output_path = output_filename.replace('.npz', f'_Heatmap.jpg')
        #image_output_path = output_filename.replace('.npz', f'_Image.jpg')
        #cv2.imwrite(overlay_output_path, heatmap_image)
        #plot_or_save_slices(heatmap_image, overlay_output_path)
        #plot_or_save_slices(d, image_output_path)
        #plot_or_save_slices(heatmap, heat_output_path)

        heatmap_LQ = generate_grad_cam(model_wrapper, input_tensor, target_class=0)
        mha_output_path = output_filename.replace('.npz', f'_LQ_heatmap.mha')

        save_heatmap_as_mha(
            heatmap_LQ,
            mha_output_path,
            preprocessed_shape=(15, 300, 300),
            original_image_sitk=original_img_sitk
        )

        heatmap_HQ = generate_grad_cam(model_wrapper, input_tensor, target_class=1)
        mha_output_path_hq = output_filename.replace('.npz', f'_HQ_heatmap.mha')
        save_heatmap_as_mha(
            heatmap_HQ,
            mha_output_path_hq,
            preprocessed_shape=(15, 300, 300),
            original_image_sitk=original_img_sitk
        )

        print("done")
