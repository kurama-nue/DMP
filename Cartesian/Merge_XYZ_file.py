#!/usr/bin/env python3
import pandas as pd
import os

# Load the three CSV files
file1 = pd.read_csv('/home/rahul/LabData/DMP-main/Cartesian_XYZ/xyz0.csv', header=None)
file2 = pd.read_csv('/home/rahul/LabData/DMP-main/Cartesian_XYZ/xyz1.csv', header=None)
file3 = pd.read_csv('/home/rahul/LabData/DMP-main/Cartesian_XYZ/xyz2.csv', header=None)

# Add a serial number column
max_length = max(len(file1), len(file2), len(file3))
serial_numbers = pd.Series(range(1, max_length + 1))

# Create a DataFrame with the serial numbers and the data from the files
merged = pd.DataFrame({
    'Serial No': serial_numbers,
    'p_x': file1[0].reindex(range(max_length)).reset_index(drop=True),
    'p_y': file2[0].reindex(range(max_length)).reset_index(drop=True),
    'p_z': file3[0].reindex(range(max_length)).reset_index(drop=True),
})

# Save the merged dataframe to a new CSV file

merged.to_csv("/home/rahul/catkin_workspace/merged_file.csv", index=False)

print("Files have been merged successfully into 'merged_file.csv'")
