from pathlib import Path
from nnunet.experiment_planning.utils import create_lists_from_splitted_dataset_folder
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import SimpleITK as sitk
import numpy as np
import json


def add_segmentations_to_task_folder(folder):
    folder = Path(folder)
    labelstr_folder = (folder / 'labelsTr')
    if (folder / 'ImagesTr').is_dir():
        imagestr_folder = (folder / 'ImagesTr')
    else:
        imagestr_folder = (folder / 'imagesTr')

    maybe_mkdir_p(str(labelstr_folder))

    image_files = create_lists_from_splitted_dataset_folder(imagestr_folder)

    filename = folder/'dataset.json'
    print(filename)
    with open(filename, 'r') as fp:
        dataset_json = json.load(fp)

    if empty_segmentations_should_be_added(dataset_json, image_files):
        print("No region of interest segmentations exist.")

        original_filename = folder/'original_dataset.json'
        print(f"Saving the original dataset.json here: {original_filename}")
        with open(original_filename, 'w') as fp:
            json.dump(dataset_json, fp, indent=2)

        print("Automatically generating region of interest segmentations with only foreground voxels...")
        add_empty_segmentation_images(image_files)

        print("Adding the region of interest segmentations to dataset.json...")
        datset_json = add_segmentations_to_dataset_json(dataset_json)

        with open(filename, 'w') as fp:
            json.dump(datset_json, fp, indent=2)
        print("Saved new dataset.json")
    else:
        print("Using existing region of interest segmentations.")



def empty_segmentations_should_be_added(dataset_json, image_files):
    tr = dataset_json["training"]

    items_dont_have_segmentations = ["label" not in tr[it].keys() for it in range(len(tr))]
    if (not all(items_dont_have_segmentations)) and any(items_dont_have_segmentations):
        raise RuntimeError("In dataset.json, only some segmentation masks are specified. "
                           "Either all or none must be specified")

    segmentation_files = corresponding_segmentation_filenames(image_files)
    seg_files_exist = [Path(s).is_file() for s in segmentation_files]

    if sum(seg_files_exist) != 0 and sum(seg_files_exist) != len(segmentation_files):
        raise RuntimeError(
            "Some segmentation masks are missing. I will generate empty masks only if all are missing. I "
            "will use the masks only if all are present.")

    if sum(seg_files_exist) == 0 or all(items_dont_have_segmentations) or "labels" not in dataset_json.keys():
        if sum(seg_files_exist) == 0 and all(items_dont_have_segmentations) and "labels" not in dataset_json.keys():
            return True
        else:
            raise RuntimeError(
                "Please make sure that either all or none of these conditions should be true:\n"
                "- No image segmentations are present in labelsTr;\n"
                "- No image segmentations are specified in dataset.json (labels field of the items in training);\n"
                "- The label field (global) is not specified in dataset.json.")
    return False


def add_dummy_modality(filename):
    return filename.replace('.nii.gz', '_0000.nii.gz')


def add_segmentations_to_dataset_json(dataset_json):
    tr = dataset_json["training"]
    for it in range(len(tr)):
        assert "label" not in tr[it].keys()
        tr[it].update({
            "label": to_corresponding_label_filename(add_dummy_modality(tr[it]['image']))
        })
    assert "labels" not in dataset_json.keys()
    dataset_json["labels"] = {
        "0": "background",
        "1": "foreground",
    }
    return dataset_json


def to_corresponding_label_filename(image_filename):
    image_filename = Path(image_filename)
    suffix = '.nii.gz'
    assert str(image_filename)[-7:] == suffix, f'images dir should only contain {suffix} image file'
    imagedir = image_filename.parent.name

    expected_folders = ['imagesTr', 'imagesTs', 'ImagesTr', 'ImagesTs']
    assert imagedir in expected_folders, f'Please make sure your images are stored in one of {expected_folders}.'
    splitted_image_filename = image_filename.name[:-len(suffix)].rsplit('_', 1)
    assert len(splitted_image_filename) == 2, "all image filenames in dataset.json must have format 'NAME_MODALITYID'"
    patient_id, modality_id = splitted_image_filename

    assert modality_id.isnumeric(), "all image filenames in dataset.json must have format 'NAME_PATIENTID_MODALITYID', where MODALITYID is numeric"

    labelsdir = imagedir.replace('image', 'label').replace("Image", "label")
    taskdir = image_filename.parent.parent
    return str(taskdir / labelsdir / (patient_id + suffix))


def corresponding_segmentation_filenames(files):
    return [to_corresponding_label_filename(f[0]) for f in files]


def add_empty_segmentation_images(image_files):
    segmentation_files = corresponding_segmentation_filenames(image_files)
    seg_files_exist = [Path(s).is_file() for s in segmentation_files]
    assert sum(seg_files_exist) == 0
    for image_file_list, segmentation_file in zip(image_files, segmentation_files):
        image_file = image_file_list[0]
        image_itk = sitk.ReadImage(image_file)
        original_size_of_raw_data = np.array(image_itk.GetSize())[[2, 1, 0]]
        segm = np.ones(original_size_of_raw_data, dtype=np.short)
        segm_itk = sitk.GetImageFromArray(segm)

        segm_itk.SetOrigin(image_itk.GetOrigin())
        segm_itk.SetSpacing(image_itk.GetSpacing())
        segm_itk.SetDirection(image_itk.GetDirection())

        sitk.WriteImage(segm_itk, str(segmentation_file), True)

