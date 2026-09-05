#!/usr/bin/env python3
"""
Script to split test.csv into 2 files based on a specified number of records.
Usage: python split_csv.py <number_of_records_for_first_file>
Example: python split_csv.py 25
"""

import pandas as pd
import sys
import os
SPLIT_SIZE = 265

def main():
    # File path
    test_file = 'auto_data_submitter/test.csv'
    
    if SPLIT_SIZE <= 0:
        print("Error: Number of records must be greater than 0.")
        sys.exit(1)
    
    print(f"Reading {test_file}...")
    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: {test_file} not found!")
        sys.exit(1)
    
    total_records = len(df)
    print(f"Total records in test.csv: {total_records}")
    
    if SPLIT_SIZE >= total_records:
        print(f"Error: Split count ({SPLIT_SIZE}) is greater than or equal to total records ({total_records}).")
        print("Please provide a number smaller than the total records.")
        sys.exit(1)
    
    # Split the dataframe
    df_first = df.head(SPLIT_SIZE)
    df_remaining = df.iloc[SPLIT_SIZE:]
    
    print(f"\nCreating first file with {len(df_first)} records...")
    print(f"Creating second file with {len(df_remaining)} records...")
    
    # Generate output filenames
    base_name = os.path.splitext(test_file)[0]
    first_file = f"{base_name}_part1.csv"
    second_file = f"{base_name}_part2.csv"
    
    # Write the split files
    df_first.to_csv(first_file, index=False)
    df_remaining.to_csv(second_file, index=False)
    
    print(f"\nSuccessfully created:")
    print(f"  {first_file}: {len(df_first)} records")
    print(f"  {second_file}: {len(df_remaining)} records")
    
    # Show sample of the first few records from each file
    print(f"\nFirst 3 records from {first_file}:")
    print(df_first.head(3).to_string(index=False))
    
    print(f"\nFirst 3 records from {second_file}:")
    print(df_remaining.head(3).to_string(index=False))

if __name__ == '__main__':
    main()