# T2W Quality Beta

## Description
This repository provides tools for performing inference on T2-weighted (T2W) medical imaging data to assess quality or generate predictions using trained models.

## Requirements
- Python 3.8 or higher is required.
- Install required Python packages using:
  ```bash
  pip install -r requirements.txt
  ```
- Bash is required for running the provided shell script.

## Folder Structure
The expected folder structure is as follows:
```
/weights_final
  ├── trained_models/ #Already provided
```

## Steps for Inference

### Preparing Input Data
- Input data should follow the naming convention: `CASENAME_XXXX.nii.gz`, where `XXXX` is the modality identifier, in our case should always be 0000.


### Running the Prediction Script


#### Arguments:
- `<INPUT_FOLDER>`: Path to the folder containing input `.nii.gz` files.
- `<DATA_PATH>`: Base path where trained model folders are located.
- `<OUTPUT_FOLDER>`: Path to save the prediction results.
- `<TASKID>`: Task ID (e.g., `Task001_ProstateQualityT2W`).
- `<FOLDER_FORMAT>`: Set to `True` if the input data is organized in folders, or `False` otherwise.


## <FOLDER_FORMAT>
 - If `True` format should be as follows:
 ```
  /patients_folder
    ├──patientid1
    │   ├──patientid1_studyid_t2w.mha #(Required)
    │   ├──patientid1_studyid_hbv.mha #(Optional)
    │   └──patientid1_studyid_adc.mha #(Optional)
    ├──patientid2
    │   ├──patientid2_studyid_t2w.mha #(Required)
    │   ├──patientid2_studyid_hbv.mha #(Optional)
    │   └──patientid2_studyid_adc.mha #(Optional)
    ...
    └──subject_list.txt
 ```
 Where the `subject_list.txt` would have the format:
 ```
 patientid1_studyid
 patientid2_studyid
 ...
 ```
 - If `False` format should be as follows:
 ```
 /patient_folder
    ├──casename1_0000.nii.gz
    ├──casename2_0000.nii.gz
    ...
    └──casenamen_0000.nii.gz
 ```
## Output

- The predictions are saved as `.npz` files in the specified `<OUTPUT_FOLDER>`. To read and process the output:
```python
import numpy as np
from scipy.special import softmax

data = np.load(path_file_npz)
logits = data['logits']
# Convert logits to probabilities using softmax
probabilities = softmax(logits, axis=-1)
# The second column contains the probabilities for the positive class
high_quality_likelihood = probabilities[..., 1]  # Use ellipsis to select all dimensions
```


## Contact
For any inquiries or issues, please contact the repository maintainer.

