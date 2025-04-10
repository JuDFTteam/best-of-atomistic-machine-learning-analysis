#!/usr/bin/env python
"""
Test script for the timeseries_extrapolation module.

This script tests the functionality to collect unique project keys and creation dates
from the historical CSV files, create an extrapolation directory with selected
CSV files, and clean up the original data for consistency.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from analysis.timeseries_extrapolation import (
    test_collect_project_keys_and_dates,
    test_create_extrapolation_directory,
    test_prepare_extrapolation_files
)


def test_clean_up_consistency(extrapolation_dir):
    """
    Test if the clean up process has made category and labels consistent across files.
    
    Args:
        extrapolation_dir: Path to the extrapolation directory
    """
    print("\nTesting category and label consistency across files...")
    
    # Get all CSV files in the extrapolation directory
    csv_files = sorted([f for f in os.listdir(extrapolation_dir) if f.endswith('.csv')])
    
    # Dictionary to store project data: {(name, homepage): [(date, category, labels), ...]}
    project_data = {}
    
    # Process each CSV file
    for file_name in tqdm(csv_files[:10], desc="Checking files for consistency"):  # Check first 10 files
        file_path = os.path.join(extrapolation_dir, file_name)
        date = file_name.split('_')[0]  # Extract date from filename
        
        try:
            df = pd.read_csv(file_path)
            
            # Process each row
            for _, row in df.iterrows():
                name = row.get('name', '')
                homepage = row.get('homepage', '')
                category = row.get('category', '')
                labels = row.get('labels', '')
                
                # Skip rows with missing name
                if pd.isna(name) or name == '':
                    continue
                    
                project_key = (name, homepage)
                
                # Store project data
                if project_key not in project_data:
                    project_data[project_key] = []
                project_data[project_key].append((date, category, labels))
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Check consistency for each project
    consistent_count = 0
    inconsistent_count = 0
    
    for project_key, data_points in project_data.items():
        if len(data_points) <= 1:
            continue  # Skip projects that appear in only one file
            
        # Check if all category and labels are the same
        categories = set(point[1] for point in data_points)
        labels_set = set(point[2] for point in data_points)
        
        if len(categories) == 1 and len(labels_set) == 1:
            consistent_count += 1
        else:
            inconsistent_count += 1
    
    print("\nConsistency check results:")
    print(f"Projects with consistent category and labels: {consistent_count}")
    print(f"Projects with inconsistent category and labels: {inconsistent_count}")
    
    if inconsistent_count == 0:
        print("All projects have consistent category and labels across files!")
    else:
        print("Some projects have inconsistent category and labels across files.")


if __name__ == "__main__":
    # Test collecting project keys and creation dates
    print("Testing collection of project keys and creation dates...")
    unique_project_keys, earliest_creation_date = test_collect_project_keys_and_dates(time_step=10)
    
    # Test creating the extrapolation directory and cleaning up original data
    print("\nTesting creation of extrapolation directory and cleaning up original data...")
    extrapolation_dir, date_sequence = test_create_extrapolation_directory(time_step=10)
    
    # Test if the clean up process has made category and labels consistent
    test_clean_up_consistency(extrapolation_dir)
    
    # Test preparing extrapolation files
    print("\nTesting preparation of extrapolation files...")
    extrapolation_files = test_prepare_extrapolation_files(
        time_step=10,
        y_property='projectrank',
        extrapolate_timesteps=1,
        extrapolate_timesteps_unit='year'
    )
    
    # Check the first and last extrapolation files
    if extrapolation_files:
        print("\nChecking contents of first extrapolation file:")
        first_file = extrapolation_files[0]
        try:
            df = pd.read_csv(first_file)
            print(f"File: {os.path.basename(first_file)}")
            print(f"Filepath: {first_file}")
            print(f"Number of projects: {len(df)}")
            print(f"Columns: {', '.join(df.columns)}")
            print(f"First few rows:\n{df.head(3)}")
        except Exception as e:
            print(f"Error reading file {first_file}: {str(e)}")
    
    print("\nTest completed successfully!")
