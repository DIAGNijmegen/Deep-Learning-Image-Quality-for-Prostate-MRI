import argparse
from universalclassifier.paths import default_plans_identifier, default_trainer
from universalclassifier.inference.predict_simple import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-s', "--seg_folder", required=False, default=None,
                        help="Folder with segmentations. If not provided, stand-in empty segmentation masks will be "
                             "generated as input to the model. If you provided segmentations at train time, please make"
                             "sure to also provide them at test time. Otherwise the model will fail silently. (TODO: "
                             "fix this in the future by saving whether real input segmentations are needed in the "
                             "trainer state)")
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default=default_plans_identifier, required=True)
    parser.add_argument('-tr', '--trainer_class_name',
                        help='Name of the trainer. The default is %s.'
                             % default_trainer,
                        required=False,
                        default=default_trainer)
    parser.add_argument('-m', '--model', help="Only 3d_fullres is currently supported. Default: 3d_fullres",
                        default="3d_fullres", required=False)
    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default=default_plans_identifier, required=False)
    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")
    parser.add_argument(
        '--folders_format',
        required=False,
        default=True,
        help="Set to True if the data is organized in folders as follows: 'data/<patient_folder>/' with individual scans "
             "for each patient stored in .mha files (e.g., '_adc.mha', '_hbv.mha', '_dwi.mha', '_t2w.mha'). "
             "The script will automatically search for '_hbv.mha' if the task name contains 'DWI' or '_t2w.mha' "
             "if the task name contains 'T2W'. If set to False, the data folder should directly contain the .nii.gz files "
             "for each sequence, pre-prepared for inference without additional folder organization."
    )
    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")
    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")
    parser.add_argument('-chk',
                        help='checkpoint name, default: model_final_checkpoint',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')

    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
