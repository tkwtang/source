#!/bin/bash

# Initialize variables
param_index=""       # Renamed from 'index' for clarity with '-i' option
num_iterations=""    # Renamed from 'number_of_loop' for clarity with '-n' option
input_script=""      # Renamed from 'input_file' to better reflect its use as a script name

# --- Argument Parsing with getopts ---
# 'f:' requires a value for input script, 'i:' for parameter index, 'n:' for number of iterations
while getopts "f:i:n:" opt; do
  case $opt in
    f)
      input_script=$OPTARG
      ;;
    i)
      param_index=$OPTARG
      ;;
    n)
      num_iterations=$OPTARG
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
if [ -z "$input_script" ]; then
  echo "Error: -f (input Python script) is a required argument." >&2
  exit 1
elif [ ! -f "$input_script" ]; then # Check if the file actually exists
    echo "Error: Input Python script '$input_script' not found." >&2
    exit 1
fi

# Validate parameter index (-i)
if [ -z "$param_index" ]; then
  echo "Error: -i (parameter index) is a required argument." >&2
  exit 1
elif ! [[ "$param_index" =~ ^[0-9]+$ ]]; then # Ensure it's a non-negative integer
  echo "Error: -i (parameter index) must be a non-negative integer." >&2
  exit 1
fi

# Validate number of iterations (-n)
if [ -z "$num_iterations" ]; then
  echo "Error: -n (number of iterations) is a required argument." >&2
  exit 1
elif ! [[ "$num_iterations" =~ ^[1-9][0-9]*$ ]]; then # Ensure it's a positive integer (no leading zeros unless it's just '0')
  echo "Error: -n (number of iterations) must be a positive integer." >&2

  exit 1
fi

# --- Main Logic ---

timestamp=$(date +%d-%m-%Y_%H-%M-%S)
# It seems you intend to copy the input script to a temporary name, then run that.
# If you just want to run the original script, remove the 'cp' and 'rm' parts.
# If copying, ensure the target directory is writable and unique name is truly needed.
TEMP_SCRIPT_NAME="placeholder/${timestamp}_$(basename "$input_script")" # Using basename for cleaner temp name
cp "$input_script" "$TEMP_SCRIPT_NAME" || { echo "Error: Failed to copy $input_script to $TEMP_SCRIPT_NAME." >&2; exit 1; }


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
    echo "Removed temporary script: $TEMP_SCRIPT_NAME"
fi

current_time=$(date +"%m-%d-%Y_%H-%M")
# $STY and $WINDOW are environment variables set by 'screen'
echo "Screen session: $STY and Window number: $WINDOW, finished running ${input_script} at $current_time" >> screen_activity.txt