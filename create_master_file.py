# create_master_file.py

"""Master Interaction File Creation Script

This script is a data preprocessing utility designed to merge pre-split RecBole
interaction files (`.train.inter`, `.valid.inter`, `.test.inter`) back into a
single, unified "master" interaction file (`.inter`).

Purpose:
--------
This is a necessary step for running certain types of RecBole experiments,
particularly those that perform their own internal data splitting (e.g., global
temporal splits using `eval_setting: 'TO_RS'`). After creating a leave-one-out
split for main benchmarks, this script can be used to reconstruct the complete,
chronologically sorted dataset that other evaluation protocols require.

Workflow:
---------
1.  Specifies an input directory containing the split files.
2.  Reads the `.train.inter`, `.valid.inter`, and `.test.inter` files into memory.
3.  Concatenates them into a single pandas DataFrame.
4.  Sorts the unified DataFrame chronologically by user and then by timestamp.
5.  Saves the final, sorted DataFrame as a single master `.inter` file in the
    same directory.

Usage:
------
The script is run directly from the command line. The input and output paths are
hardcoded as constants at the top of the file and should be modified as needed.

    python create_master_file.py

Dependencies:
-------------
- pandas
- os
- glob
"""
import pandas as pd
import os

# Path to the directory containing the split files
input_dir = 'dataset/b2b_data_fdsa'
output_file = os.path.join(input_dir, 'b2b_data_fdsa.inter')

# List of files to merge
files_to_merge = [
    os.path.join(input_dir, 'b2b_data_fdsa.train.inter'),
    #dataset\b2b_data\b2b_data.test.inter
    os.path.join(input_dir, 'b2b_data_fdsa.valid.inter'),
    os.path.join(input_dir, 'b2b_data_fdsa.test.inter')
]

print("Starting to merge files to create the master .inter file...")

# Read and concatenate the files
df_list = [pd.read_csv(f, sep='\t') for f in files_to_merge]
full_df = pd.concat(df_list, ignore_index=True)

# Sort the master file for consistency
full_df.sort_values(by=['user_id:token', 'timestamp:float'], inplace=True)

# Save the master file
full_df.to_csv(output_file, index=False, sep='\t')

print(f"Success! Created master file: {output_file} with {len(full_df)} interactions.")