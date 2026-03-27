
import shutil
import nnunet.utilities.shutil_sol as shutil_sol
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p
from nnunet.configuration import default_num_threads
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from nnunet.experiment_planning.utils import create_lists_from_splitted_dataset

from universalclassifier.preprocessing.cropping import ClassificationImageCropper


def crop(task_string, override=False, num_threads=default_num_threads, create_dummy_seg=False):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ClassificationImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil_sol.copyfile(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)