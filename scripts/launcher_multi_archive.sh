#!/usr/bin/env bash

# Serialize the command and its arguments into a string
export EXEC_COMMAND=$(printf "%q " "$@")

script="$1"
scriptbase="`basename "$script"`"
scriptbase="${scriptbase%.py}"

name="$scriptbase"_`date +%Y-%m-%d_%H-%M-%S`
dir=archive/"$name"

echo will mkdir -p "$dir"
mkdir -p "$dir"

echo will cp -r *.py configs scripts "$dir"
cp -r *.py configs scripts "$dir"

cd "$dir"
chmod -R g+w .

# Execute the saved command using exec
echo will sbatch "$EXEC_COMMAND" in "$dir"
echo "$EXEC_COMMAND" > cmd.sh
exec sbatch -a 1-10%1 --comment="$EXEC_COMMAND" $(dirname "$0")/launcher_job.sh

