#!/usr/bin/env python3
"""
Script to remove duplicate records from test.csv that are present in Form_responses.csv
Matches only on the "Name" column.
"""

import pandas as pd
import sys

def main():
    # File paths
    test_file = 'auto_data_submitter/test.csv'
    form_responses_file = 'auto_data_submitter/Form_responses.csv'
    
    print(f"Reading {test_file}...")
    try:
        test_df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: {test_file} not found!")
        sys.exit(1)
    
    print(f"Reading {form_responses_file}...")
    try:
        form_df = pd.read_csv(form_responses_file)
    except FileNotFoundError:
        print(f"Error: {form_responses_file} not found!")
        sys.exit(1)
    
    print(f"\ntest.csv: {len(test_df)} records")
    print(f"Form_responses.csv: {len(form_df)} records")
    
    # Standardize column names for comparison
    test_df.columns = test_df.columns.str.strip()
    form_df.columns = form_df.columns.str.strip()
    
    # Key column to match on
    key_column = 'Name'
    
    if key_column not in test_df.columns:
        print(f"Error: '{key_column}' column not found in {test_file}")
        sys.exit(1)
    if key_column not in form_df.columns:
        print(f"Error: '{key_column}' column not found in {form_responses_file}")
        sys.exit(1)
    
    # Strip whitespace from string columns in both dataframes
    for col in test_df.select_dtypes(include=['object', 'string']).columns:
        test_df[col] = test_df[col].astype(str).str.strip()
    
    for col in form_df.select_dtypes(include=['object', 'string']).columns:
        form_df[col] = form_df[col].astype(str).str.strip()

    # Normalize the Name column itself for matching (case-insensitive, trimmed)
    test_df['_name_key'] = test_df[key_column].str.strip().str.lower()
    form_df['_name_key'] = form_df[key_column].str.strip().str.lower()
    
    # Find names in test.csv that are already present in Form_responses.csv
    test_names = set(test_df['_name_key'])
    form_names = set(form_df['_name_key'])
    
    duplicate_names = test_names & form_names
    
    print(f"\nDuplicate names found (present in Form_responses.csv): {len(duplicate_names)}")
    
    # Filter test_df to keep only records whose Name isn't in Form_responses.csv
    test_df_cleaned = test_df[~test_df['_name_key'].isin(duplicate_names)].copy()
    
    # Also drop duplicate names WITHIN test.csv itself, keeping the first occurrence
    before_internal = len(test_df_cleaned)
    test_df_cleaned = test_df_cleaned.drop_duplicates(subset='_name_key', keep='first')
    internal_dupes_removed = before_internal - len(test_df_cleaned)
    
    # Remove the temporary key column
    test_df_cleaned = test_df_cleaned.drop('_name_key', axis=1)
    
    print(f"\nOriginal test.csv: {len(test_df)} records")
    print(f"Cleaned test.csv: {len(test_df_cleaned)} records")
    print(f"Removed (matched Form_responses.csv): {len(test_df) - before_internal}")
    print(f"Removed (internal duplicates within test.csv): {internal_dupes_removed}")
    print(f"Total removed: {len(test_df) - len(test_df_cleaned)} records")
    
    # Write the cleaned data back to test.csv
    test_df_cleaned.to_csv(test_file, index=False)
    print(f"\nSuccessfully updated {test_file}")
    
    # Show some examples of removed duplicates if any
    if len(duplicate_names) > 0:
        print("\nSample of names removed (matched against Form_responses.csv):")
        sample = test_df[test_df['_name_key'].isin(duplicate_names)][[key_column]].drop_duplicates().head(10)
        print(sample.to_string(index=False))

if __name__ == '__main__':
    main()