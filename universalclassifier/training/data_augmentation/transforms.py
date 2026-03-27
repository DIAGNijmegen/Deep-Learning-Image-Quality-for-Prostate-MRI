from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class RescaleSegmentationTransform(AbstractTransform):
    '''
    Rescales the segmentation at input_key that is the range [min_label, max_label] to the range
    [min_output, max_output]. It saves its results as float in output_key
    '''

    def __init__(self, max_label, min_label=0, max_output=1., min_output=0., input_key="seg", output_key="seg"):
        self.max_label = max_label
        self.min_label = min_label
        self.max_output = max_output
        self.min_output = min_output
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        seg = seg.astype(np.float32)
        seg = (seg - self.min_label) / (self.max_label - self.min_label)
        seg = seg * (self.max_output - self.min_output) + self.min_output
        data_dict[self.output_key] = seg
        return data_dict
