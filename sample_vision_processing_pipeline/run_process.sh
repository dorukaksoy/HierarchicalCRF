#!/bin/bash

# Check for 'img.tif'
if [[ ! -f "img.tif" ]]; then
  echo "Error: 'img.tif' not found. Please rename/reformat the image to be processed to 'img.tif'."
  exit 1
fi

echo "Do you want to use the Human-in-the-Loop (HITL) approach? (Y/n)"
read HITL_response
HITL_response=$(echo "$HITL_response" | tr '[:upper:]' '[:lower:]')

if [[ $HITL_response == "yes" || $HITL_response == "y" ]]; then
  HITL=True
else
  HITL=False
fi

if [[ $HITL == "True" ]]; then
  echo "The HITL approach allows for manual correction of segmentation paths. Please review 'all_viable_paths.png' and 'crf_predicted_paths.png' for segment indices."
  echo "Broken or completion segments can be added/removed from the 'crf_predicted_paths.png' using four csv files: broken_to_add.csv, broken_to_remove.csv, completion_to_add.csv, completion_to_remove.csv."
  echo "These contain the segment indices obtained from 'all_viable_paths.png' separated by commas."

  # Check for existing CSV files
  csv_files_exist=false
  for file in broken_to_add.csv broken_to_remove.csv completion_to_add.csv completion_to_remove.csv; do
    if [[ -f "$file" ]]; then
      csv_files_exist=true
      break
    fi
  done

  if [[ $csv_files_exist == "true" ]]; then
    echo "CSV files from a previous run exist. Do you want to clear these files? (Y/n)"
    read clear_all_response
    clear_all_response=$(echo "$clear_all_response" | tr '[:upper:]' '[:lower:]')
    if [[ $clear_all_response == "yes" || $clear_all_response == "y" ]]; then
      for file in broken_to_add.csv broken_to_remove.csv completion_to_add.csv completion_to_remove.csv; do
        > "$file"  # Clear the file content
        echo "$file cleared."
      done
    else
      echo "Keeping all CSV files."
    fi
  fi
else
  # If HITL is not used, clear the files directly
  for file in broken_to_add.csv broken_to_remove.csv completion_to_add.csv completion_to_remove.csv; do
    if [[ -f "$file" ]]; then
      > "$file"  # Clear the file content
      echo "$file cleared."
    fi
  done
fi

# Run initial scripts
scripts=(Post0_UNET_VGG16.py Post1_PixelBasedCRF.py Post2_Classifier.py)
if [[ $HITL == "True" ]]; then
  scripts+=("Post3_SegmentBasedCRF.py True")
else
  scripts+=("Post3_SegmentBasedCRF.py False")
fi

for script in "${scripts[@]}"; do
  python $script
  if [[ $? -ne 0 ]]; then
    echo "Error encountered running $script. Stopping."
    exit 1
  fi
done

if [[ $HITL == "True" ]]; then
  echo "Please review 'all_viable_paths.png' and 'crf_predicted_paths.png'."
  echo "Then, update the CSV files as needed. These can be empty but should follow the instructions provided earlier."
  echo "Press enter once you've updated the CSV files."
  read
fi

# Continue with the rest of the scripts if no errors
if [[ $HITL == "True" ]]; then
  remaining_scripts=("Post4_Pathfinding.py True")
else
  scripts=("Post4_Pathfinding.py False")
fi
remaining_scripts+=(Post5_Segmentation.py Post6_LineScanPoints.py)

for script in "${remaining_scripts[@]}"; do
  echo "Running $script..."
  python $script
  if [[ $? -ne 0 ]]; then
    echo "Error encountered running $script. Stopping."
    exit 1
  fi
done

echo "All processes are completed."