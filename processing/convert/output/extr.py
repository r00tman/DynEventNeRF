#!/usr/bin/env python3
# Extract only needed RGB frames (<frames_file>) and views ('_4.png') from training data RGB folder
import os
import shutil
import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 4:
    print("Usage: python script.py <source_directory> <destination_directory> <frames_file>")
    sys.exit(1)

# Define the source and destination directories from command line arguments
source_directory = sys.argv[1]
destination_directory = sys.argv[2]
frames_file = sys.argv[3]

# Read the frames from the specified text file
try:
    with open(frames_file, 'r') as f:
        frames = [line.strip() for line in f if line.strip().isdigit()]
except FileNotFoundError:
    print(f"Error: The file '{frames_file}' was not found.")
    sys.exit(1)

# Iterate through the specified frames
for frame in frames:
    folder = f"{int(frame):04d}"  # Format the frame number as a 4-digit string
    folder_path = os.path.join(source_directory, folder)

    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate through the files in the folder
        for file in os.listdir(folder_path):
            # Check if the file ends with '_4.png'
            if file.endswith('_4.png'):
                # Construct the new file name
                new_file_name = f"{frame}.png"
                # Define the source file path
                source_file_path = os.path.join(folder_path, file)
                # Define the destination file path
                destination_file_path = os.path.join(destination_directory, new_file_name)

                # Copy the file to the destination with the new name
                shutil.copy(source_file_path, destination_file_path)
                print(f"Copied: {source_file_path} to {destination_file_path}")

print("File copying completed.")

