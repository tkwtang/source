#!/bin/bash

# Initialize variables
num_iterations=""
sid=""

# --- Argument Parsing with getopts ---
# 's:' requires an argument for the script name
# 'n:' requires an argument for the number of iterations
while getopts "s:n:" opt; do
  case $opt in
    s)
      sid=$OPTARG
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

# Validate input script file (-s)
if [ -z "$sid" ]; then
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

timestamp=$(date +%d-%m-%Y_%H-%M-%S)
# It seems you intend to copy the input script to a temporary name, then run that.
# If you just want to run the original script, remove the 'cp' and 'rm' parts.
# If copying, ensure the target directory is writable and unique name is truly needed.
TEMP_SCRIPT_NAME="placeholder/${timestamp}_dummy_simulator.py" # Using basename for cleaner temp name
cp "dummy_simulator.py" "$TEMP_SCRIPT_NAME" || { echo "Error: Failed to copy dummy_simulator.py to $TEMP_SCRIPT_NAME." >&2; exit 1; }


# --- Main Logic ---
echo "SID = $sid"
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
    python "dummy_simulator.py" "$sid"
done


# --- Final Cleanup and Logging ---
if [ -f "$TEMP_SCRIPT_NAME" ]; then
    rm -f "$TEMP_SCRIPT_NAME" # Clean up the temporary script after loop completes
    echo "Removed temporary script: $TEMP_SCRIPT_NAME"
fi
