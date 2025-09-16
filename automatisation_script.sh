#!/bin/bash

# Directory containing ZIP files
ZIP_DIR="/path/to/my_export"

# Python scripts to run
PREPROCESS_SCRIPT="scripts/preprocess_neon.py"
SEGMENT_SCRIPT="scripts/segmentation.py"

# Directory to save results
RESULT_DIR="results"

# Folders to delete after each run
TO_DELETE=("temp")

for zip_file in "$ZIP_DIR"/*.zip; do
    echo "Processing: $zip_file"
    
    # Step 1: Preprocessing
    python "$PREPROCESS_SCRIPT" --zip_path "$zip_file"
    if [ $? -ne 0 ]; then
        echo "Preprocessing failed for $zip_file"
        continue
    fi

    # Step 2: Segmentation
    python "$SEGMENT_SCRIPT" --result_path "$RESULT_DIR"
    if [ $? -ne 0 ]; then
        echo "Segmentation failed for $zip_file"
        continue
    fi

    # Clean up
    for dir in "${TO_DELETE[@]}"; do
        if [ -d "$dir" ]; then
            echo "Deleting folder: $dir"
            rm -rf "$dir"
        else
            echo "Folder $dir not found, nothing to delete."
        fi
    done

    echo "Processing finished for: $zip_file"
    echo "----------------------------"
done