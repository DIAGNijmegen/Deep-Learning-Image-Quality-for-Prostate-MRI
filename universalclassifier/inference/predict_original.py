import shutil
import nnunet.utilities.shutil_sol as shutil_sol

from typing import Tuple, Union, List

from copy import deepcopy
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import torch
from universalclassifier.training.model_restore import load_model_and_checkpoint_files
from universalclassifier.inference.export import save_output


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

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


def predict_from_folder(model: str, input_folder: str, seg_folder: str, output_folder: str,
                        folds: Union[Tuple[int], List[int]],
                        mixed_precision: bool = True, overwrite_existing: bool = True,
                        checkpoint_name: str = "model_final_checkpoint"):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases
    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :param checkpoint_name:
    :return:
    """
    maybe_mkdir_p(output_folder)
    shutil_sol.copyfile(join(model, 'plans.pkl'), output_folder)

    assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

    output_files = [join(output_folder, i + ".npz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
    if seg_folder is None:
        print("No segmentation folder specified. Will generate emtpy segmentation masks for inference.")
        seg_files = [None] * len(case_ids)
    else:
        seg_files = subfiles(seg_folder, suffix=".nii.gz", join=False, sort=True)
        expected_segfiles = [os.path.basename(d[0]).rsplit('_')[0] + '.nii.gz' for d in list_of_lists]
        if not all([f in seg_files for f in expected_segfiles]):
            raise RuntimeError(f"Please make sure that for each case ID in {input_folder} a corresponding case ID "
                               f"exists in {seg_folder}.")
        seg_files = [join(seg_folder, f) for f in expected_segfiles]
        print(list_of_lists)
        print(seg_files)
    return predict_cases(model, list_of_lists, seg_files, output_files, folds, mixed_precision=mixed_precision,
                         overwrite_existing=overwrite_existing, checkpoint_name=checkpoint_name)


def predict_cases(model, list_of_lists_of_modality_filenames, seg_filenames, output_filenames, folds,
                  mixed_precision=True, overwrite_existing=False, checkpoint_name="model_final_checkpoint"):
    """
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists_of_modality_filenames: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param seg_filenames: [case0.nii.gz, case1.nii.gz, ...], or [None, None, ...] if the model was trained with emtpy segmentation masks
    :param output_filenames: [output_file_case0.npz, output_file_case1.npz, ...]
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param overwrite_existing: default: True
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :param checkpoint_name:
    :return:
    """
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
        print(f"PREDDDDDD: {pred}")
        for p in params[1:]:
            trainer.load_checkpoint_ram(p, False)
            new_pred = trainer.predict_preprocessed_data_return_pred_and_logits(d[None], mixed_precision=mixed_precision)[1]
            for it in range(len(pred)):
                pred = [p + n_p for p, n_p in zip(pred, new_pred)]

        if len(params) > 1:
            pred = [p/len(params) for p in pred]

        print("exporting prediction...")
        categorical_output = [np.argmax(p) for p in pred]
        save_output(categorical_output, pred, output_filename, properties)
        print("done")

