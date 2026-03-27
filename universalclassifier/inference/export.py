
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

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def save_output(pred_categorical: List[np.ndarray],
                pred_softmax: List[np.ndarray],
                pred_softmax_npz_fname: str,
                properties_dict: dict):
    if pred_softmax_npz_fname is not None:
        np.savez_compressed(pred_softmax_npz_fname,
                            logits=pred_softmax,
                            categorical=pred_categorical)
        save_pickle(properties_dict, pred_softmax_npz_fname[:-4] + ".pkl")


