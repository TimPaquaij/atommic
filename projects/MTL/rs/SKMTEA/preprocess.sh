#!/bin/bash
echo "
Preprocessing pipeline for the Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) 2021 Dataset.

For more information, please refer to https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7
and check the following paper https://openreview.net/forum?id=YDMFgD_qJuA.

Generating train, val, and test sets...
"

# Prompt the user to enter the path to the downloaded annotations directory
echo "Please enter the data directory to raw data:"
read INPUT_DIR

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "The input directory does not exist. Please try again."
  exit 1
fi

# Prompt the user to enter the path to the downloaded annotations directory
echo "Please enter the annotations:"
read INPUT_DIR_AN

# Check if the input directory exists
if [ ! -d "$INPUT_DIR_AN" ]; then
  echo "The input directory does not exist. Please try again."
  exit 1
fi

# Run the json generation script
python projects/MTL/rs/SKMTEA/scripts/preprocess_data.py $INPUT_DIR $INPUT_DIR_AN --data_type raw
echo "Done!"
