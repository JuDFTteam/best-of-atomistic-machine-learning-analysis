"""
Time Series Extrapolation Module for Atomistic Machine Learning Projects

This module provides functionality to extrapolate project data backwards to their creation dates,
creating a synthesized dataset that can be used with the timeseries_visualization module.
"""

import os
import re
import logging
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Set

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesExtrapolator:
    """
    Class for extrapolating time series data backwards to project creation dates.
    
    This class handles the creation of a synthesized dataset that combines original data
    with extrapolated data going back to each project's creation date.
    """
    
    def __init__(
        self, 
        history_dir: str = None,
    ):
        """
        Initialize the TimeSeriesExtrapolator.
        
        Args:
            history_dir: Path to the directory containing historical CSV files
        """
        self.history_dir = history_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "analysis", "history"
        )
        
        # Validate paths
        if not os.path.exists(self.history_dir):
            raise FileNotFoundError(f"History directory not found: {self.history_dir}")
        
        # Initialize data structures
        self.csv_files = []
        self.unique_project_keys = set()
        self.original_region_date_sequence = []
        self.extrapolation_dir = os.path.join(self.history_dir, "extrapolation_data")
        self.earliest_csv_date = None
        self.latest_csv_date = None
        self.project_first_appearance = {}
        self.logger = logging.getLogger(__name__)
        
    def clean_up_original_data(self):
        """
        Clean up original data in the extrapolation directory.
        For each project, find the first CSV file it appears in, extract category and labels,
        and ensure consistency across all files.
        
        Returns:
            dict: Dictionary mapping project keys to their first appearance info (date, category, labels).
        """
        self.logger.info("Cleaning up original data in the extrapolation directory")
        
        if not self.extrapolation_dir or not os.path.exists(self.extrapolation_dir):
            self.logger.error("Extrapolation directory does not exist. Create it first with create_extrapolation_directory()")
            return {}
        
        # Dictionary to store project first appearance info: {(name, homepage): (first_date, category, labels)}
        project_first_appearance = {}
        
        # Dictionary to store all project data: {(name, homepage): {date: row_data}}
        all_project_data = {}
        
        # First pass: Find the first appearance of each project and extract category and labels
        for date in tqdm(sorted(self.original_region_date_sequence), desc="Finding first appearance of each project"):
            file_name = date.strftime("%Y-%m-%d") + "_projects.csv"
            file_path = os.path.join(self.extrapolation_dir, file_name)
            
            if not os.path.exists(file_path):
                continue
                
            try:
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                required_columns = ['name', 'homepage', 'category', 'labels']
                if not all(col in df.columns for col in required_columns):
                    self.logger.warning(f"Skipping file {file_path} due to missing required columns")
                    continue
                    
                # Process each row
                for _, row in df.iterrows():
                    name = row.get('name', '')
                    homepage = row.get('homepage', '')
                    
                    # Skip rows with missing name
                    if pd.isna(name) or name == '':
                        continue
                        
                    project_key = (name, homepage)
                    
                    # Store project data for all dates
                    if project_key not in all_project_data:
                        all_project_data[project_key] = {}
                    all_project_data[project_key][date] = row.to_dict()
                    
                    # If this is the first appearance of the project, store its info
                    if project_key not in project_first_appearance:
                        category = row.get('category', '')
                        labels = row.get('labels', '')
                        project_first_appearance[project_key] = (date, category, labels)
                        
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {str(e)}")
        
        self.logger.info(f"Found first appearance info for {len(project_first_appearance)} projects")
        
        # Second pass: Ensure consistency of category and labels across all files
        modified_files_count = 0
        for date in tqdm(sorted(self.original_region_date_sequence), desc="Ensuring consistency across files"):
            file_name = date.strftime("%Y-%m-%d") + "_projects.csv"
            file_path = os.path.join(self.extrapolation_dir, file_name)
            
            if not os.path.exists(file_path):
                continue
                
            try:
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                required_columns = ['name', 'homepage', 'category', 'labels']
                if not all(col in df.columns for col in required_columns):
                    self.logger.warning(f"Skipping file {file_path} due to missing required columns")
                    continue
                
                # Create a dictionary of existing projects in this file
                existing_projects = {}
                for idx, row in df.iterrows():
                    name = row.get('name', '')
                    homepage = row.get('homepage', '')
                    
                    if pd.isna(name) or name == '':
                        continue
                        
                    existing_projects[(name, homepage)] = idx
                
                # Update existing projects with consistent category and labels
                modified = False
                for project_key, (first_date, category, labels) in project_first_appearance.items():
                    if project_key in existing_projects:
                        idx = existing_projects[project_key]
                        
                        # Update category and labels if they're different
                        if df.at[idx, 'category'] != category or df.at[idx, 'labels'] != labels:
                            df.at[idx, 'category'] = category
                            df.at[idx, 'labels'] = labels
                            modified = True
                
                # Save the modified DataFrame back to the file if changes were made
                if modified:
                    df.to_csv(file_path, index=False)
                    modified_files_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {str(e)}")
        
        self.logger.info(f"Modified {modified_files_count} files to ensure category and label consistency")
        self.logger.info("Completed cleaning up original data in the extrapolation directory")
        
        # Store the project first appearance info for later use
        self.project_first_appearance = project_first_appearance
        
        return project_first_appearance
    
    def _find_csv_files(self, time_step: int = 1) -> Tuple[List[str], List[datetime]]:
        """
        Find all CSV files in the history directory matching the pattern YYYY-MM-DD_projects.csv.
        
        Args:
            time_step: Sampling frequency for CSV files
            
        Returns:
            Tuple containing:
            - List of file paths sorted by date
            - List of dates corresponding to the files (original_region_date_sequence)
        """
        # Pattern to match YYYY-MM-DD_projects.csv
        pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_projects\.csv$')
        
        # Find all matching files
        all_files = []
        for file in os.listdir(self.history_dir):
            match = pattern.search(file)
            if match:
                date_str = match.group(1)
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    all_files.append((date, os.path.join(self.history_dir, file)))
                except ValueError:
                    logger.warning(f"Could not parse date from filename: {file}")
        
        # Sort files by date
        all_files.sort(key=lambda x: x[0])
        
        # Store the earliest and latest CSV dates
        if all_files:
            self.earliest_csv_date = all_files[0][0]
            self.latest_csv_date = all_files[-1][0]
        
        # Create the original_region_date_sequence
        original_region_date_sequence = []
        
        # Sample files based on time_step
        if time_step > 1 and len(all_files) > 2:
            # Always include first and last file
            sampled_files = [all_files[0][1]]
            original_region_date_sequence.append(all_files[0][0])
            
            # Sample intermediate files
            for i in range(1, len(all_files) - 1, time_step):
                sampled_files.append(all_files[i][1])
                original_region_date_sequence.append(all_files[i][0])
            
            # Add last file if not already included
            if all_files[-1][1] not in sampled_files:
                sampled_files.append(all_files[-1][1])
                original_region_date_sequence.append(all_files[-1][0])
                
            return sampled_files, original_region_date_sequence
        
        # Return all files if time_step is 1 or there are too few files
        return [file[1] for file in all_files], [file[0] for file in all_files]
    
    def _extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Extract date from filename in format YYYY-MM-DD_projects.csv.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Datetime object or None if date cannot be extracted
        """
        pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_projects\.csv$')
        match = pattern.search(os.path.basename(filename))
        
        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                logger.warning(f"Could not parse date from filename: {filename}")
        
        return None
    
    def _parse_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse a CSV file and return its contents as a DataFrame.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the CSV data
        """
        try:
            df = pd.read_csv(file_path, low_memory=False)
            return df
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}")
            return pd.DataFrame()
    
    def collect_project_keys_and_dates(self, time_step: int = 1) -> Tuple[Set[Tuple[str, str]], datetime]:
        """
        Collect unique project keys and creation dates from all CSV files.
        
        Args:
            time_step: Sampling frequency for CSV files
            
        Returns:
            Tuple containing the set of unique project keys and the earliest creation date
        """
        # Find CSV files and get the original_region_date_sequence
        self.csv_files, self.original_region_date_sequence = self._find_csv_files(time_step)
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {self.history_dir}")
        
        logger.info(f"Processing {len(self.csv_files)} CSV files to collect project keys and dates")
        
        # Initialize sets
        self.unique_project_keys = set()
        self.created_dates = set()
        
        # Process each file
        for file_path in tqdm(self.csv_files, desc="Collecting project keys and dates"):
            # Parse CSV file
            df = self._parse_csv_file(file_path)
            if df.empty:
                logger.warning(f"Skipping empty or invalid file: {file_path}")
                continue
            
            # Check if required columns exist
            if 'name' not in df.columns or 'homepage' not in df.columns:
                missing_cols = []
                if 'name' not in df.columns:
                    missing_cols.append('name')
                if 'homepage' not in df.columns:
                    missing_cols.append('homepage')
                logger.warning(f"Missing required columns in {file_path}: {', '.join(missing_cols)}")
                continue
            
            # Collect unique project keys (name, homepage)
            for _, row in df.iterrows():
                name = row.get('name', '')
                homepage = row.get('homepage', '')
                
                # Skip rows with missing name
                if pd.isna(name) or name == '':
                    continue
                
                # Handle missing homepage
                if pd.isna(homepage):
                    homepage = ''
                
                # Add to set of unique project keys
                self.unique_project_keys.add((name, homepage))
                
                # Collect creation date if available
                created_at = row.get('created_at', None)
                if created_at and not pd.isna(created_at):
                    try:
                        # Convert to datetime if it's a string
                        if isinstance(created_at, str):
                            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        else:
                            created_date = created_at
                        
                        self.created_dates.add(created_date)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse creation date '{created_at}' for project {name}: {e}")
        
        # Get the earliest creation date
        if self.created_dates:
            self.earliest_creation_date = min(self.created_dates)
            logger.info(f"Found {len(self.unique_project_keys)} unique projects")
            logger.info(f"Earliest project creation date: {self.earliest_creation_date}")
            logger.info(f"Earliest CSV file date: {self.earliest_csv_date}")
            logger.info(f"Latest CSV file date: {self.latest_csv_date}")
            logger.info(f"Original region date sequence contains {len(self.original_region_date_sequence)} dates")
        else:
            logger.warning("No valid creation dates found in any project")
            self.earliest_creation_date = None
        
        return self.unique_project_keys, self.earliest_creation_date


def create_extrapolation_directory(
    history_dir: Optional[str] = None, 
    time_step: int = 1
) -> Tuple[str, List[datetime]]:
    """
    Create a temporary directory for extrapolation data and copy selected CSV files.
    
    Args:
        history_dir: Path to the directory containing historical CSV files
        time_step: Sampling frequency for CSV files
        
    Returns:
        Tuple containing:
        - Path to the extrapolation directory
        - List of dates in the original region date sequence
    """
    extrapolator = TimeSeriesExtrapolator(history_dir=history_dir)
    
    # Collect project keys and dates
    extrapolator.collect_project_keys_and_dates(time_step=time_step)
    
    # Create extrapolation directory
    if os.path.exists(extrapolator.extrapolation_dir):
        logger.info(f"Removing existing extrapolation directory: {extrapolator.extrapolation_dir}")
        shutil.rmtree(extrapolator.extrapolation_dir)
    
    os.makedirs(extrapolator.extrapolation_dir, exist_ok=True)
    logger.info(f"Created extrapolation directory: {extrapolator.extrapolation_dir}")
    
    # Copy selected CSV files to the extrapolation directory
    for file_path in tqdm(extrapolator.csv_files, desc="Copying CSV files to extrapolation directory"):
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(extrapolator.extrapolation_dir, file_name)
        shutil.copy2(file_path, dest_path)
    
    logger.info(f"Copied {len(extrapolator.csv_files)} CSV files to extrapolation directory")
    
    # Clean up original data in the extrapolation directory
    extrapolator.clean_up_original_data()
    
    return extrapolator.extrapolation_dir, extrapolator.original_region_date_sequence


def test_collect_project_keys_and_dates(history_dir: Optional[str] = None, time_step: int = 1):
    """
    Test function to collect and display unique project keys and the earliest creation date.
{{ ... }}
    Args:
        history_dir: Path to the directory containing historical CSV files
        time_step: Sampling frequency for CSV files
    """
    extrapolator = TimeSeriesExtrapolator(history_dir=history_dir)
    
    unique_project_keys, earliest_creation_date = extrapolator.collect_project_keys_and_dates(time_step=time_step)
    
    print(f"Number of unique projects: {len(unique_project_keys)}")
    print(f"Earliest project creation date: {earliest_creation_date}")
    print(f"Earliest CSV file date: {extrapolator.earliest_csv_date}")
    print(f"Latest CSV file date: {extrapolator.latest_csv_date}")
    print(f"Number of dates in original region sequence: {len(extrapolator.original_region_date_sequence)}")
    
    # Print a few example project keys
    print("\nExample project keys:")
    for i, key in enumerate(list(unique_project_keys)[:5]):
        print(f"{i+1}. Name: {key[0]}, Homepage: {key[1]}")
    
    return unique_project_keys, earliest_creation_date


def test_create_extrapolation_directory(history_dir: Optional[str] = None, time_step: int = 1):
    """
    Test function to create the extrapolation directory and copy selected CSV files.
    
    Args:
        history_dir: Path to the directory containing historical CSV files
        time_step: Sampling frequency for CSV files
    """
    extrapolation_dir, date_sequence = create_extrapolation_directory(history_dir=history_dir, time_step=time_step)
    
    print(f"Extrapolation directory created at: {extrapolation_dir}")
    print(f"Number of dates in original region sequence: {len(date_sequence)}")
    
    # Print the first few and last few dates in the sequence
    if len(date_sequence) > 0:
        print("\nFirst few dates in sequence:")
        for i, date in enumerate(date_sequence[:min(5, len(date_sequence))]):
            print(f"{i+1}. {date.strftime('%Y-%m-%d')}")
        
        if len(date_sequence) > 5:
            print("\nLast few dates in sequence:")
            for i, date in enumerate(date_sequence[-min(5, len(date_sequence)):]):
                print(f"{len(date_sequence)-min(5, len(date_sequence))+i+1}. {date.strftime('%Y-%m-%d')}")
    
    # Count files in the extrapolation directory
    file_count = len([f for f in os.listdir(extrapolation_dir) if f.endswith('.csv')])
    print(f"\nNumber of CSV files in extrapolation directory: {file_count}")
    
    return extrapolation_dir, date_sequence


if __name__ == "__main__":
    # Run the test functions
    print("Testing collection of project keys and creation dates...")
    test_collect_project_keys_and_dates(time_step=10)
    
    print("\nTesting creation of extrapolation directory...")
    test_create_extrapolation_directory(time_step=10)
