#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from universalclassifier.inference.predict import predict_from_folder
from nnunet.paths import network_training_output_dir
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def predict(args):
    input_folder = args.input_folder
    seg_folder = args.seg_folder
    output_folder = args.output_folder
    folds = args.folds
    overwrite_existing = args.overwrite_existing
    model = args.model
    trainer_class_name = args.trainer_class_name
    folders_format = args.folders_format.lower() == "true"
    task_name = args.task_name



    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    if 'DWI' in task_name:
        modality = 'DWI'
    elif 'T2W' in task_name:
        modality = 'T2W'
    else:
        print('No modality found in Task Name')

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

    trainer = trainer_class_name

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
                             args.plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    if seg_folder is not None and isdir(seg_folder):
        print("\nSegmentation folder found. If this folder does not contain cases for each case in the input folder, "
              "I will crash. If training was done without segmentation masks, please abort and rerun without specifying"
              " the segmentation folder.\n")
    else:
        print("\nSegmentation folder not found. I will generate empty segmentation masks to runn inference. \nWARNING: If "
              "training was done with segmentation masks, I will run inference, but fail silently, producing "
              "poor results.\n")  #Todo make it not fail silently

    predict_from_folder(model_folder_name, input_folder, output_folder, folds,
                        not args.disable_mixed_precision, overwrite_existing, args.chk, folders_format, modality)
