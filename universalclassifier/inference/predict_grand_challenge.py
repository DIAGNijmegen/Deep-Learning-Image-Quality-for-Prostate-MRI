import os
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from typing import Tuple, Union, List
from scipy.special import softmax
from universalclassifier.paths import default_plans_identifier, default_trainer
from universalclassifier.training.model_restore import load_model_and_checkpoint_files


def predict_grand_challenge(artifact_path: str,
                            ordered_image_files: List[str],
                            roi_segmentation_file: str = None,
                            folds: Union[Tuple[int], List[int]] = None,
                            model: str = "3d_fullres",
                            trainer_class_name: str = default_trainer,
                            plans_identifier: str = default_plans_identifier,
                            disable_mixed_precision: bool = True,
                            checkpoint_name: str = "model_final_checkpoint"):
    mixed_precision = not disable_mixed_precision

    if folds is None:
        folds = ["all"]

    task_path = os.path.join(artifact_path, "nnUNet", "3d_fullres")
    task_names = os.listdir(task_path)
    if len(task_names) == 0:
        raise RuntimeError(f"No artifacts found in {task_path}")
    if len(task_names) > 1:
        print(f"Expected artifacts for only one task, but found artifacts for: {task_names}. Please only provide the artifact of a single task to limit the size of the docker container.")
    task_name = task_names[0]

    if not task_name.startswith("Task"):
        raise RuntimeError("task_name should start with 'Task'")

    assert model in ["3d_fullres"], "-m must be 3d_fullres"

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    model_folder_name = join(artifact_path, "nnUNet", model, task_name, trainer_class_name + "__" +
                             plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    expected_num_modalities = load_pickle(join(model_folder_name, "plans.pkl"))['num_modalities']
    assert len(ordered_image_files) == expected_num_modalities, \
        f"Expected {expected_num_modalities} input modalities (excluding the optional roi segmentation), but len(ordered_image_files)=={len(ordered_image_files)}"

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model_folder_name, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)

    print(f"=== Processing {ordered_image_files}, {roi_segmentation_file}:")
    print("preprocessing...")
    d, s, properties = trainer.preprocess_patient(ordered_image_files, roi_segmentation_file)
    d = trainer.combine_data_and_seg(d, s)

    print("predicting...")
    trainer.load_checkpoint_ram(params[0], False)
    pred = trainer.predict_preprocessed_data_return_pred_and_logits(d[None], mixed_precision=mixed_precision)[1]
    pred = [softmax(p) for p in pred]

    for p in params[1:]:
        trainer.load_checkpoint_ram(p, False)
        new_pred = trainer.predict_preprocessed_data_return_pred_and_logits(d[None], mixed_precision=mixed_precision)[1]
        pred = [p + softmax(n_p) for p, n_p in zip(pred, new_pred)]

    if len(params) > 1:
        pred = [p / len(params) for p in pred]

    pred = [p[0].tolist() for p in pred]  # remove batch dimension and convert to list for storing as json

    # Binary task outputs need to be stored as the probability of the positive class
    pred = [p if len(p) > 2 else p[1] for p in pred]

    return pred
