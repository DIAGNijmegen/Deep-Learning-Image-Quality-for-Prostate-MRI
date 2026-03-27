from batchgenerators.utilities.file_and_folder_operations import *


def summarize_plans(file):
    plans = load_pickle(file)
    print("num_classes in region of interest segmentation: ", plans['num_classes'])
    print("num_classes for classification: ", plans['num_classification_classes'])
    print("classification labels: ", plans['all_classification_labels'])
    print("modalities: ", plans['modalities'])
    print("use_mask_for_norm", plans['use_mask_for_norm'])
    print("normalization_schemes", plans['normalization_schemes'])
    print("stages...\n")

    for i in range(len(plans['plans_per_stage'])):
        print("stage: ", i)
        print(plans['plans_per_stage'][i])
        print("")