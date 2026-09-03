#!/usr/bin/env python3
"""
Script to remove duplicate records from test.csv that are present in Form_responses.csv
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
    
    # Define key columns to match on (normalized column names)
    # test.csv columns: Name, Gender, Age Group, Povince, Cancer Type, etc.
    # Form_responses.csv columns: Timestamp, Name, Gender, Age Group, Povince, Cancer Type, etc.
    
    # Standardize column names for comparison
    test_df.columns = test_df.columns.str.strip()
    form_df.columns = form_df.columns.str.strip()
    
    # Key columns to match on
    key_columns = ['Name', 'Gender', 'Age Group', 'Povince', 'Cancer Type']
    
    # Check if key columns exist in both dataframes
    test_cols = set(test_df.columns)
    form_cols = set(form_df.columns)
    
    # Normalize Form_responses columns - remove extra spaces and fix column names
    form_df = form_df.rename(columns={
        'Povince': 'Povince',  # Already matches
        '  Cancer Type  ': 'Cancer Type',
        '  Duration Since Diagnosis  ': 'Duration Since Diagnosis',
        '  Current Cancer Status  ': 'Current Cancer Status',
        '  Current Treatment Type  ': 'Current Treatment Type'
    })
    
    # Strip whitespace from string columns in both dataframes
    for col in test_df.select_dtypes(include=['object', 'string']).columns:
        test_df[col] = test_df[col].astype(str).str.strip()
    
    for col in form_df.select_dtypes(include=['object', 'string']).columns:
        form_df[col] = form_df[col].astype(str).str.strip()
    
    # Create a composite key for matching
    test_df['match_key'] = test_df[key_columns].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
    form_df['match_key'] = form_df[key_columns].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
    
    # Find records in test.csv that are NOT in Form_responses.csv
    test_keys = set(test_df['match_key'])
    form_keys = set(form_df['match_key'])
    
    duplicates = test_keys & form_keys
    unique_keys = test_keys - form_keys
    
    print(f"\nDuplicate records found: {len(duplicates)}")
    print(f"Unique records to keep: {len(unique_keys)}")
    
    # Filter test_df to keep only unique records
    test_df_cleaned = test_df[~test_df['match_key'].isin(duplicates)].copy()
    
    # Remove the temporary match_key column
    test_df_cleaned = test_df_cleaned.drop('match_key', axis=1)
    
    print(f"\nOriginal test.csv: {len(test_df)} records")
    print(f"Cleaned test.csv: {len(test_df_cleaned)} records")
    print(f"Removed: {len(test_df) - len(test_df_cleaned)} duplicate records")
    
    # Write the cleaned data back to test.csv
    test_df_cleaned.to_csv(test_file, index=False)
    print(f"\nSuccessfully updated {test_file}")
    
    # Show some examples of removed duplicates if any
    if len(duplicates) > 0:
        print("\nSample of removed duplicate records:")
        sample_duplicates = test_df[test_df['match_key'].isin(duplicates)][key_columns].head(5)
        print(sample_duplicates.to_string(index=False))

if __name__ == '__main__':
    main()