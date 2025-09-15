#!/bin/bash

# Initialize variables
num_iterations=""
json_location=""

# --- Argument Parsing with getopts ---
# 's:' requires an argument for the script name
# 'n:' requires an argument for the number of iterations
while getopts "s:n:" opt; do
  case $opt in
    s)
      json_location=$OPTARG
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

# Validate input json file (-s)
if [ -z "$json_location" ]; then
  echo "Error: -s (input Python script) is a required argument." >&2
  exit 1
fi

# Validate number of iterations (-n)
if [ -z "$num_iterations" ]; then
  echo "Error: -n (number of iterations) is a required argument." >&2
  exit 1
elif ! [[ "$num_iterations" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: -n (number of iterations) must be a positive integer." >&2
  exit 1
fi


# It seems you intend to copy the input script to a temporary name, then run that.
# If you just want to run the original script, remove the 'cp' and 'rm' parts.
# If copying, ensure the target directory is writable and unique name is truly needed.
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
TEMP_SCRIPT_NAME="placeholder/${timestamp}_dummy_simulator_json_input.py" # Using basename for cleaner temp name
cp "dummy_simulator_json_input.py" "$TEMP_SCRIPT_NAME" || { echo "Error: Failed to copy $input_script to $TEMP_SCRIPT_NAME." >&2; exit 1; }


# --- Main Logic ---
echo "json_location = $json_location"
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
    # Execute the Python script with the index and script name
    python "dummy_simulator_json_input.py" "$json_location"
done


# --- Final Cleanup and Logging ---
if [ -f "$TEMP_SCRIPT_NAME" ]; then
    rm -f "$TEMP_SCRIPT_NAME" # Clean up the temporary script after loop completes
    echo "Removed temporary script: $TEMP_SCRIPT_NAME"
fi


current_time=$(date +"%m-%d-%Y_%H-%M")
# $STY and $WINDOW are environment variables set by 'screen'
echo "Screen session: $STY and Window number: $WINDOW, finished running ${json_location} at $current_time" >> screen_activity.txt
