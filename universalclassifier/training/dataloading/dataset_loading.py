from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import get_case_identifiers
import numpy as np

def load_dataset(folder_with_preprocessed_data, dataset_directory, cases_key, num_cases_properties_loading_threshold=1000):
    dataset_json = load_json(join(dataset_directory, 'dataset.json'))
    cases = dataset_json[cases_key]
    classification_labels = dataset_json['classification_labels']
    metadata = {os.path.basename(case['image']).replace(".nii.gz", ""): case for case in cases}

    # we don't load the actual data but instead return the filename to the np file.
    case_identifiers = get_case_identifiers(folder_with_preprocessed_data)
    case_identifiers.sort()
    print(len(set(metadata.keys())))
    print(len(set(case_identifiers)))
    assert set(metadata.keys()) == set(case_identifiers)

    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder_with_preprocessed_data, "%s.npz" % c)

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]['properties_file'] = join(folder_with_preprocessed_data, "%s.pkl" % c)
        dataset[c]['classification_labels'] = classification_labels
        if cases_key == 'training':
            dataset[c]['target'] = np.asarray([metadata[c][label["name"]] for label in classification_labels])

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder_with_preprocessed_data, "%s_segs.npz" % c)

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset
