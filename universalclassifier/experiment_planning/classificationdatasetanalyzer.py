from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

from collections import OrderedDict
from multiprocessing import Pool


class ClassificationDatasetAnalyzer(DatasetAnalyzer):

    # Added upper bound of 1e9 voxels to compute statistics from. This should be enough to get a good statistic.
    # ROI might be smaller, but probably not a lot smaller since we are cropping to it
    def _get_voxels_in_foreground(self, patient_identifier, modality_id):
        print(f"Collecting foreground voxels for {patient_identifier}...")
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data']
        modality = all_data[modality_id]
        mask = all_data[-1] > 0

        voxels = list(modality[mask][::self.foreground_voxel_stride])  # edit: Changed 10 to foreground_voxel_stride
        return voxels

    def set_foreground_voxel_stride(self, nr_voxels_upper_bound):
        voxels_in_dataset = np.sum([np.product(s) for s in self.sizes])
        self.foreground_voxel_stride = int(np.ceil(voxels_in_dataset / nr_voxels_upper_bound))
        self.foreground_voxel_stride = max(10, self.foreground_voxel_stride)

    def analyze_dataset(self, collect_intensityproperties=True, nr_voxels_upper_bound=1e9):
        # get all spacings and sizes
        self.sizes, self.spacings = self.get_sizes_and_spacings_after_cropping()
        self.set_foreground_voxel_stride(nr_voxels_upper_bound)

        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        classes = self.get_classes()
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]

        classification_labels = self.get_classification_labels()
        all_classification_labels = [[int(i) for i in label["values"].keys()] for label in classification_labels]

        # modalities
        modalities = self.get_modalities()

        # collect intensity information
        if collect_intensityproperties:
            # self._set_total_nr_voxels_in_foreground()
            intensityproperties = self.collect_intensity_properties(len(modalities))
        else:
            intensityproperties = None

        print("intensity properties collected.")

        # size reduction by cropping
        size_reductions = self.get_size_reduction_by_cropping()

        dataset_properties = dict()
        dataset_properties['all_sizes'] = self.sizes
        dataset_properties['all_spacings'] = self.spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['all_classification_labels'] = all_classification_labels
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        dimensions = [[si * sp for si, sp in zip(size, spacing)] for size, spacing in zip(self.sizes, self.spacings)]
        dataset_properties['max_dimensions'] = [np.max(d) for d in zip(*dimensions)]  # added

        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties

    def get_classification_labels(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        return datasetjson['classification_labels']

    # Below here, I made no changes except for print statements to make running jobs more verbose
    def collect_intensity_properties(self, num_modalities):
        if self.overwrite or not isfile(self.intensityproperties_file):
            p = Pool(self.num_processes)

            results = OrderedDict()
            for mod_id in range(num_modalities):
                results[mod_id] = OrderedDict()
                print(f"Modality {mod_id}: Collecting foreground voxels...")
                v = p.starmap(self._get_voxels_in_foreground, zip(self.patient_identifiers,
                                                                  [mod_id] * len(self.patient_identifiers)))

                print(f"Modality {mod_id}: Concatenating foreground voxel results...")
                w = []
                for iv in v:
                    w += iv
                print(f"Modality {mod_id}: Computing global stats over foreground voxels of all images...")
                median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._compute_stats(w)

                print(f"Modality {mod_id}: Computing stats over foreground voxels of each image individually...")
                local_props = p.map(self._compute_stats, v)
                props_per_case = OrderedDict()
                print(f"Modality {mod_id}: Saving stats...")
                for i, pat in enumerate(self.patient_identifiers):
                    props_per_case[pat] = OrderedDict()
                    props_per_case[pat]['median'] = local_props[i][0]
                    props_per_case[pat]['mean'] = local_props[i][1]
                    props_per_case[pat]['sd'] = local_props[i][2]
                    props_per_case[pat]['mn'] = local_props[i][3]
                    props_per_case[pat]['mx'] = local_props[i][4]
                    props_per_case[pat]['percentile_99_5'] = local_props[i][5]
                    props_per_case[pat]['percentile_00_5'] = local_props[i][6]

                results[mod_id]['local_props'] = props_per_case
                results[mod_id]['median'] = median
                results[mod_id]['mean'] = mean
                results[mod_id]['sd'] = sd
                results[mod_id]['mn'] = mn
                results[mod_id]['mx'] = mx
                results[mod_id]['percentile_99_5'] = percentile_99_5
                results[mod_id]['percentile_00_5'] = percentile_00_5

            p.close()
            p.join()
            save_pickle(results, self.intensityproperties_file)
        else:
            results = load_pickle(self.intensityproperties_file)
        return results

    @staticmethod
    def _compute_stats(voxels):
        print(f"Computing stats over {len(voxels)} voxels...")
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5
