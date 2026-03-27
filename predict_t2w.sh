#!/bin/bash

# Define hardcoded arguments for the Python script
INPUT_FOLDER="/path/to/input/files"
DATA_PATH="./weights_t2w" #Not need to change
OUTPUT_FOLDER="/path/to/output/folder"
TASKID="Task001_ProstateQualityT2W" #Not need to change
FOLDS=0 #Variable not used. Thus all folds are used #Not need to change
FOLDER_FORMAT="False" #Explained in the README 

if ! [ -d "$DATA_PATH/trained_models" ]; then
  echo "Please make sure that $DATA_PATH/trained_models/ exists"
  exit
fi

export RESULTS_FOLDER="$DATA_PATH/trained_models" #Not need to change


# Run the Python script with the specified arguments
echo "Running python3 -u uc_predict.py with predefined arguments"
python3 -u uc_predict.py \
  -i $INPUT_FOLDER \
  -o $OUTPUT_FOLDER \
  -t $TASKID \
  --folders_format $FOLDER_FORMAT
