#!/bin/bash

placeholder_name="placeholder"
sid_file="$placeholder_name/thermalize_id_for_Experiment 1 (2025-08-27).txt"
comment=""
timestamp=$(date +%d-%m-%Y_%H-%M-%S)
batch_size=500

# --- Argument Parsing with getopts ---
# 'f:' requires a value for input script, 'i:' for parameter index, 'n:' for number of iterations
while getopts "f:c:n:" opt; do
  case $opt in
    f)
      sid_file=$OPTARG
      ;;
    c)
      comment=$OPTARG
      ;;
    n)
      batch_size=$OPTARG
      ;;
    \?) # Handle invalid options
      echo "Error: Invalid option -$OPTARG" >&2
      exit 1
      ;;
    :) # Handle options missing an argument
      echo "Error: Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Shift processed options from the positional parameters ($@)
shift $((OPTIND-1))

# --- Argument Validation ---
# Validate input script file (-f)
if [ -z "$sid_file" ]; then
  echo "Error: -f (input Python script) is a required argument." >&2
  exit 1
elif [ ! -f "$sid_file" ]; then # Check if the file actually exists
    echo "Error: Input Python script '$sid_file' not found." >&2
    exit 1
fi

# Validate comment (-c)
if [ -z "$comment" ]; then
  echo "Error: -c comment is a required argument." >&2
  exit 1
fi


# Validate number of iterations (-n)
if [ -z "$batch_size" ]; then
  echo "Error: -n (number of iterations) is a required argument." >&2
  exit 1
elif ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then # Ensure it's a positive integer (no leading zeros unless it's just '0')
  echo "Error: -n (number of iterations) must be a positive integer." >&2

  exit 1
fi

# --- Main Logic ---

# Initialize variables
# Set variables
batch_file_name="$placeholder_name/$timestamp"+"_batch_sid.txt"
python_script_name="thermalize_final_state_using_json_input.py"
echo batch_file_name

TEMP_SCRIPT_NAME="${placeholder_name}/${timestamp}_${python_script_name}" # Using basename for cleaner temp name
cp "$python_script_name" "$TEMP_SCRIPT_NAME" || { echo "Error: Failed to copy $input_script to $TEMP_SCRIPT_NAME." >&2; exit 1; }


echo "Input Python script: $input_script"
echo "Parameter index: $param_index"
echo "Number of iterations: $num_iterations"
echo "All required arguments provided and validated. Script executing..."
echo "Temporary script filename: $TEMP_SCRIPT_NAME"

# --- Interrupt Handler ---
cleanup_on_interrupt() {
    echo -e "\nCtrl+C detected! Performing cleanup..."
    if [ -f "$TEMP_SCRIPT_NAME" ]; then # Only remove if it exists
        rm -f "$TEMP_SCRIPT_NAME" # Use -f to force removal without prompt
        echo "Removed temporary script: $TEMP_SCRIPT_NAME"
    fi
    echo "Cleanup complete. Exiting."
    exit 1 # Exit with a non-zero status to indicate abnormal termination
}


# Process the file in a loop
head -n "$batch_size" "$sid_file" > "$batch_file_name" 

# Remove the processed lines from the original file
tail -n +"$((batch_size + 1))" "$sid_file" > "temp_file.txt" && mv "temp_file.txt" "$sid_file"

head -n "$batch_size" "$batch_file_name" | while IFS= read -r line; do
    python "$python_script_name" "$line" "$comment"
    tail -n +2 "$batch_file_name" > "temp_file.txt" && mv "temp_file.txt" "$batch_file_name"
done






# Trap SIGINT (Ctrl+C) and call the cleanup_on_interrupt function
trap cleanup_on_interrupt SIGINT

# --- Loop Execution ---
for (( i=0; i < num_iterations; i++ ))
do
    # Execute the copied Python script with the index argument
    python "$TEMP_SCRIPT_NAME" "$param_index"
    # echo "Loop iteration $((i+1)): python \"$TEMP_SCRIPT_NAME\" \"$param_index\"" # For testing/debugging
done

# --- Final Cleanup and Logging ---
if [ -f "$TEMP_SCRIPT_NAME" ]; then
    rm -f "$TEMP_SCRIPT_NAME" # Clean up the temporary script after loop completes
    rm $batch_file_name
    echo "Removed temporary script: $TEMP_SCRIPT_NAME"
fi

current_time=$(date +"%m-%d-%Y_%H-%M")
# $STY and $WINDOW are environment variables set by 'screen'
echo "Screen session: $STY and Window number: $WINDOW, finished running ${input_script} at $current_time" >> screen_activity.txt