import numpy as np
import pickle
import os
from nnunet.configuration import default_num_threads
from nnunet.preprocessing.cropping import get_case_identifier_from_npz
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing.pool import Pool
from nnunet.preprocessing.preprocessing import GenericPreprocessor
from universalclassifier.preprocessing.cropping import ClassificationImageCropper
from universalclassifier.preprocessing.padding import central_pad  # Import padding function

class UniversalClassifierPreprocessor(GenericPreprocessor):

    def run(self, target_spacings, target_sizes, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        """
        Runs the preprocessing pipeline for a single stage with fixed spacing and size.

        Args:
            target_spacings (list of lists): Desired voxel spacings for the single stage, e.g., [[0.5, 0.5, 3]].
            target_sizes (list of lists): Desired output sizes for the single stage, e.g., [[20, 512, 512]].
            input_folder_with_cropped_npz (str): Path to the folder containing cropped NPZ files.
            output_folder (str): Path to the folder where preprocessed data will be saved.
            data_identifier (str): Identifier for the dataset.
            num_threads (int, optional): Number of threads for multiprocessing. Defaults to default_num_threads.
            force_separate_z (bool, optional): Parameter for handling separate z-axis if needed. Defaults to None.
        """
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)

        # Get list of files to preprocess
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        maybe_mkdir_p(output_folder)

        # Single stage setup
        spacing = target_spacings
        target_size = target_sizes
        output_folder_stage = os.path.join(output_folder, f"{data_identifier}_stage0")
        maybe_mkdir_p(output_folder_stage)

        # Prepare arguments for multiprocessing
        all_args = []
        for case in list_of_cropped_npz_files:
            case_identifier = get_case_identifier_from_npz(case)
            args = (
            spacing, target_size, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z)
            all_args.append(args)

        # Run preprocessing in parallel with the specified number of threads
        with Pool(num_threads) as p:
            p.starmap(self._run_internal, all_args)

    def _run_internal(self, target_spacing, target_size, case_identifier, output_folder_stage, cropped_output_dir,
                      force_separate_z):
        """
        Internal method to handle preprocessing of a single case, including resampling and padding.

        Args:
            target_spacing (list or tuple): Desired voxel spacing [x, y, z].
            target_size (list or tuple): Desired output size [x, y, z].
            case_identifier (str): Identifier for the case.
            output_folder_stage (str): Path to the output folder for the current stage.
            cropped_output_dir (str): Path to the cropped data directory.
            force_separate_z (bool, optional): Parameter for handling separate z-axis if needed.
        """
        # Load the data from the cropped directory
        data, seg, properties = self.load_cropped(cropped_output_dir, case_identifier)


        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(data, target_spacing,
                                                            properties, seg, force_separate_z)

        data, seg, properties = central_pad(data, target_size, properties, seg)

        # Save the preprocessed data
        all_data = np.vstack((data, seg)).astype(np.float32)
        np.savez_compressed(
            os.path.join(output_folder_stage, f"{case_identifier}.npz"),
            data=all_data
        )
        with open(os.path.join(output_folder_stage, f"{case_identifier}.pkl"), 'wb') as f:
            pickle.dump(properties, f)

    def preprocess_test_case(self, data_files, target_spacing, target_size, seg_file=None, force_separate_z=None):
        """
        Preprocesses a single test case with fixed spacing and size.

        Args:
            data_files (list): List of paths to image files.
            seg_file (str, optional): Path to the segmentation file. Defaults to None.
            force_separate_z (bool, optional): Parameter for handling separate z-axis if needed. Defaults to None.

        Returns:
            tuple: Preprocessed data, segmentation, and updated properties.
        """
        # Load and crop the data
        data, seg, properties = ClassificationImageCropper.crop_from_list_of_files(
            data_files, seg_file, create_dummy_seg=(seg_file is None)
        )

        # Apply transpose for consistent orientation
        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        # Resample and normalize the data to the fixed target spacing
        data, seg, properties = self.resample_and_normalize(
            data, target_spacing, properties, seg, force_separate_z
        )

        # Apply central padding to match the fixed target size
        data, seg, properties = central_pad(data, target_size, properties, seg)

        return data.astype(np.float32), seg, properties
