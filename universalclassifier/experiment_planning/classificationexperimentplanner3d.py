
import shutil
import nnunet.utilities.shutil_sol as shutil_sol

import nnunet
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.configuration import default_num_threads
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
import numpy as np

import universalclassifier


class ClassificationExperimentPlanner3D(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "universal_classifier_plans_v1.0"
        self.plans_fname = join(self.preprocessed_output_folder, "UniversalClassifierPlansv1.0_plans_3D.pkl")
        self.preprocessor_name = "UniversalClassifierPreprocessor"
        self.max_shape_limit = [240, 300, 300]  # hard coded for now
        self.minimum_batch_size = 5  # Works for I3dr model with 240^3 img size
        self.fixed_spacing = [3, .3, .3]  # Desired spacing in mm
        self.fixed_size = [15, 300, 300]  # Desired image size in voxels

    def get_properties_for_stage(self, current_spacing, image_size):
        """
        Set fixed properties for a given stage, ignoring original spacing and max shape.

        Args:
            original_spacing (array): Original image spacing (ignored).
            max_shape (array): Original max shape of the image (ignored).

        Returns:
            dict: Dictionary containing fixed properties for the stage.
        """

        plan = {
            'batch_size': self.minimum_batch_size,
            'image_size': image_size,
            'current_spacing': current_spacing,
            'do_dummy_2D_data_aug': False,
        }
        return plan

    def plan_experiment(self):
        # Determine whether to use a nonzero mask for normalization
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero mask for normalization?", use_nonzero_mask_for_normalization)

        all_classes = self.dataset_properties['all_classes']
        all_classification_labels = self.dataset_properties['all_classification_labels']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        # Use the fixed spacing and size defined for the quality classifier
        target_spacing = np.array(self.fixed_spacing)
        target_size = self.fixed_size

        # Define transpose parameters
        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in range(3) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [self.transpose_forward.index(i) for i in range(3)]

        # Logging fixed values for reference
        print("Fixed target spacing:", target_spacing)
        print("Fixed target size:", target_size)

        # Plan configuration for a single 3D stage
        self.plans_per_stage = {
            0: self.get_properties_for_stage(target_spacing, target_size)
        }

        print("Generating configuration for 3d_fullres with fixed target spacing and size")
        print("Transpose forward:", self.transpose_forward)
        print("Transpose backward:", self.transpose_backward)

        # Determine normalization scheme
        normalization_schemes = self.determine_normalization_scheme()

        # Create the overall plan
        plans = {
            'num_stages': 1,
            'num_modalities': num_modalities,
            'modalities': modalities,
            'normalization_schemes': normalization_schemes,
            'dataset_properties': self.dataset_properties,
            'list_of_npz_files': self.list_of_cropped_npz_files,
            'original_spacings': [self.fixed_spacing],
            'original_sizes': [self.fixed_size],
            'preprocessed_data_folder': self.preprocessed_output_folder,
            'num_classes': len(all_classes),
            'all_classes': all_classes,
            'num_classification_classes': [len(labels) for labels in all_classification_labels],
            'all_classification_labels': all_classification_labels,
            'use_mask_for_norm': use_nonzero_mask_for_normalization,
            'transpose_forward': self.transpose_forward,
            'transpose_backward': self.transpose_backward,
            'data_identifier': self.data_identifier,
            'plans_per_stage': self.plans_per_stage,
            'preprocessor_name': self.preprocessor_name,
        }

        self.plans = plans
        self.save_my_plans()

    def run_preprocessing(self, num_threads):
        # Remove any existing ground truth segmentations if present
        gt_seg_path = join(self.preprocessed_output_folder, "gt_segmentations")
        if os.path.isdir(gt_seg_path):
            shutil.rmtree(gt_seg_path)
        shutil_sol.copytree(join(self.folder_with_cropped_data, "gt_segmentations"), gt_seg_path)

        # Fetch parameters from the plans
        normalization_schemes = self.plans['normalization_schemes']
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties']['intensityproperties']

        # Load the preprocessor class
        preprocessor_class = recursive_find_python_class(
            [join(universalclassifier.__path__[0], "preprocessing")],
            self.preprocessor_name,
            current_module="universalclassifier.preprocessing"
        )
        assert preprocessor_class is not None

        # Initialize the preprocessor
        preprocessor = preprocessor_class(
            normalization_schemes,
            use_nonzero_mask_for_normalization,
            self.transpose_forward,
            intensityproperties
        )

        # Use the fixed target spacing and size for preprocessing
        target_spacing = self.fixed_spacing
        target_size = self.fixed_size

        # Adjust num_threads for single stage
        if isinstance(num_threads, (list, tuple)):
            num_threads = num_threads[-1]

        # Run the preprocessor
        preprocessor.run(
            target_spacing,
            target_size,
            self.folder_with_cropped_data,
            self.preprocessed_output_folder,
            self.plans['data_identifier'],
            num_threads
        )