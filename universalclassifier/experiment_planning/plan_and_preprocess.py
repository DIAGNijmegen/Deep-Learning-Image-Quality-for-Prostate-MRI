# Altered from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py



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


from batchgenerators.utilities.file_and_folder_operations import *
from universalclassifier.experiment_planning.utils import crop

import shutil
import nnunet.utilities.shutil_sol as shutil_sol
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity as verify_dataset_integrity_original_function
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.paths import *

from universalclassifier.preprocessing import utils
from universalclassifier.experiment_planning.classificationdatasetanalyzer import ClassificationDatasetAnalyzer
import universalclassifier
import nnunet


def find_planner(search_in, planner_name, current_module):
    """Finds and returns the specified 3D planner class."""
    planner = recursive_find_python_class([search_in], planner_name, current_module=current_module)
    if planner is None:
        raise RuntimeError(f"Could not find Planner class {planner_name} in {current_module}")
    return planner

def main(args):
    task_id = int(args.task_id)  # Single task only
    task_name = convert_id_to_task_name(task_id)
    raw_folder = join(nnUNet_raw_data, task_name)

    utils.add_segmentations_to_task_folder(raw_folder)
    # Verify dataset integrity if requested
    if args.verify_dataset_integrity:
        print("Verifying dataset integrity...", flush=True)
        verify_dataset_integrity_original_function(raw_folder)

    # Crop data
    print("Cropping...", flush=True)
    crop(task_name, False, args.tf)

    # Set up 3D planner
    planner_3d = find_planner(
        search_in=join(universalclassifier.__path__[0], 'experiment_planning'),
        planner_name=args.planner3d,
        current_module="universalclassifier.experiment_planning"
    )

    # Output directory setup
    cropped_out_dir = join(nnUNet_cropped_data, task_name)
    preprocessing_output_dir_task = join(preprocessing_output_dir, task_name)
    maybe_mkdir_p(preprocessing_output_dir_task)

    # Dataset Analysis
    print("Analyzing data...", flush=True)
    dataset_analyzer = ClassificationDatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=args.tf)
    dataset_analyzer.analyze_dataset(collect_intensityproperties=True)  # Assume MRI intensity properties needed

    # Copy necessary files
    shutil_sol.copyfile(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_task)
    shutil_sol.copyfile(join(nnUNet_raw_data, task_name, "dataset.json"), preprocessing_output_dir_task)

    # Execute 3D planner
    print("Planning and preprocessing...", flush=True)
    exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_task)
    exp_planner.plan_experiment()
    if not args.no_pp:
        exp_planner.run_preprocessing((args.tl, args.tf))

if __name__ == "__main__":
    main()