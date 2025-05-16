#!/bin/bash

API_1=""
API_2=""
API_3=""
API_4=""
API_5=""
API_6=""
API_7=""
API_8=""

PYTHON_SCRIPT=" process_designs.py" 
DESIGN_FOLDER="LS-flatten-500" 

BASE_OUTPUT_DIR="516_results" 


mkdir -p "$BASE_OUTPUT_DIR"

INSTANCE_OUTPUT_FOLDER0="${BASE_OUTPUT_DIR}/valid_0"
mkdir -p "$INSTANCE_OUTPUT_FOLDER0"

INSTANCE_OUTPUT_FOLDER1="${BASE_OUTPUT_DIR}/valid_1"
mkdir -p "$INSTANCE_OUTPUT_FOLDER1"

INSTANCE_OUTPUT_FOLDER2="${BASE_OUTPUT_DIR}/valid_2"
mkdir -p "$INSTANCE_OUTPUT_FOLDER2"

INSTANCE_OUTPUT_FOLDER3="${BASE_OUTPUT_DIR}/valid_3"
mkdir -p "$INSTANCE_OUTPUT_FOLDER3"

INSTANCE_OUTPUT_FOLDER4="${BASE_OUTPUT_DIR}/valid_4"
mkdir -p "$INSTANCE_OUTPUT_FOLDER4"

INSTANCE_OUTPUT_FOLDER5="${BASE_OUTPUT_DIR}/valid_5"
mkdir -p "$INSTANCE_OUTPUT_FOLDER5"

INSTANCE_OUTPUT_FOLDER6="${BASE_OUTPUT_DIR}/valid_6"
mkdir -p "$INSTANCE_OUTPUT_FOLDER6"

INSTANCE_OUTPUT_FOLDER7="${BASE_OUTPUT_DIR}/valid_7"
mkdir -p "$INSTANCE_OUTPUT_FOLDER7"





python "$PYTHON_SCRIPT" \
    --design_folder "$DESIGN_FOLDER" \
    --design_names_file "valid_0.json" \
    --output_folder "$INSTANCE_OUTPUT_FOLDER0" \
    --api_key "$API_1" &

python "$PYTHON_SCRIPT" \
    --design_folder "$DESIGN_FOLDER" \
    --design_names_file "valid_1.json" \
    --output_folder "$INSTANCE_OUTPUT_FOLDER1" \
    --api_key "$API_2" &

python "$PYTHON_SCRIPT" \
    --design_folder "$DESIGN_FOLDER" \
    --design_names_file "valid_2.json" \
    --output_folder "$INSTANCE_OUTPUT_FOLDER2" \
    --api_key "$API_3" &

python "$PYTHON_SCRIPT" \
    --design_folder "$DESIGN_FOLDER" \
    --design_names_file "valid_3.json" \
    --output_folder "$INSTANCE_OUTPUT_FOLDER3" \
    --api_key "$API_4" &

python "$PYTHON_SCRIPT" \
    --design_folder "$DESIGN_FOLDER" \
    --design_names_file "valid_4.json" \
    --output_folder "$INSTANCE_OUTPUT_FOLDER4" \
    --api_key "$API_5" &

python "$PYTHON_SCRIPT" \
    --design_folder "$DESIGN_FOLDER" \
    --design_names_file "valid_5.json" \
    --output_folder "$INSTANCE_OUTPUT_FOLDER5" \
    --api_key "$API_6" &

python "$PYTHON_SCRIPT" \
    --design_folder "$DESIGN_FOLDER" \
    --design_names_file "valid_6.json" \
    --output_folder "$INSTANCE_OUTPUT_FOLDER6" \
    --api_key "$API_7" &

python "$PYTHON_SCRIPT" \
    --design_folder "$DESIGN_FOLDER" \
    --design_names_file "valid_7.json" \
    --output_folder "$INSTANCE_OUTPUT_FOLDER7" \
    --api_key "$API_8" &


wait