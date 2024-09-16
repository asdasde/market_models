#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -n <number_of_sessions> -d <input_directory>"
    exit 1
}

# Initialize variables
NUM_SESSIONS=""
INPUT_DIR=""

# Parse the command line arguments
while getopts "n:d:" opt; do
    case $opt in
        n)
            NUM_SESSIONS=$OPTARG
            ;;
        d)
            INPUT_DIR=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

# Check if the required arguments are provided
if [ -z "$NUM_SESSIONS" ] || [ -z "$INPUT_DIR" ]; then
    usage
fi

# Set the base directory to the directory where the script is located
BASE_DIR=$(dirname "$0")

# Get the base name of the input directory for screen session naming
BASE_NAME=$(basename "$INPUT_DIR")

echo "Input parameters:"
echo "Number of sessions: $NUM_SESSIONS"
echo "Input directory: $INPUT_DIR"

# Loop to start the specified number of screen sessions
for ((i=1; i<=NUM_SESSIONS; i++)); do
    screen -dmS "${BASE_NAME}_crawler_$i" python3 "$BASE_DIR/crawler_old.py" "$INPUT_DIR"
    echo "Started screen session ${BASE_NAME}_crawler_$i"

    # Add a 5-second delay, but not after the last session
    if [ $i -lt $NUM_SESSIONS ]; then
        echo "Waiting 5 seconds before starting the next session..."
        sleep 5
    fi
done

echo "All sessions have been started."