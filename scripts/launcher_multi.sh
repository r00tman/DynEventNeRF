#!/usr/bin/env bash

# Serialize the command and its arguments into a string
export EXEC_COMMAND=$(printf "%q " "$@")

# Execute the saved command using exec
exec sbatch -a 1-5%1 --comment="$EXEC_COMMAND" $(dirname "$0")/launcher_job.sh

