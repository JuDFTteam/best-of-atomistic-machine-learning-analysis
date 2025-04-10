# Timeseries Extrapolation for Atomistic Machine Learning Projects

## Project Overview

This document outlines the requirements and design for creating a time series extrapolation module that extends project data backwards to their creation dates, complementing the existing time series visualization functionality for the "best-of-atomistic-machine-learning" repository.

### Data Source
- CSV files in `analysis/history` following the pattern `YYYY-MM-DD_projects.csv`
- Files span from 2023-06-09 to 2025-04-09 (approximately 100 files)
- Each file contains metadata about atomistic machine learning projects
- Project creation dates are stored in the `created_at` column
- The earliest project creation date is approximately 2010-05-11, about 13 years before the first CSV file

### Extrapolation Goal
Create a synthesized dataset that combines original data with extrapolated data points going back to each project's creation date. This extended dataset can then be used with the existing visualization module to show a more complete picture of the evolution of project metrics over time.

## Functional Requirements

### Core Functionality
1. Identify unique projects and their creation dates from the original CSV files
2. Create a temporary workspace with selected original CSV files
3. Generate extrapolated data points for each project between its creation date and first appearance
4. Combine original and extrapolated data into a synthesized dataset
5. Provide a visual indicator (vertical line) to distinguish between extrapolated and actual data

### Input Parameters
- **y_property**: Dynamic property to extrapolate (default: `projectrank`)
- **time_step**: Sampling frequency for CSV files (default: `1` - use all files)
- **history_dir**: Path to the directory containing historical CSV files
- **extrapolate_timesteps**: Number of timesteps to use for extrapolation (default: `1`)
- **extrapolate_timesteps_unit**: Unit for extrapolation timesteps (default: `year`, options: `day`, `month`, `year`)
- **extrapolate_keep_data**: Whether to keep the extrapolation data after processing (default: `False`)

### Data Handling
- Projects with missing `created_at` values will be excluded from extrapolation
- Projects with missing values for the selected `y_property` will be excluded from extrapolation
- Primary key for projects: tuple of `name` and `homepage` columns
- Original data region: time range between the first and last CSV files [y, z]
- Extrapolation data region: time range between the earliest project creation date and the first CSV file [x, y]

## Technical Design

### Processing Pipeline
1. **Collection Phase**
   - Create a date sequence for the original data date range [y, z]
   - Scan directory for all CSV files matching a date pattern from the sequence (always include first and last)
   - Collect unique project keys (name, homepage) and creation dates
   - Identify the earliest project creation date

2. **Workspace Preparation**
   - Create a temporary extrapolation directory in which we'll work exclusively
   - Copy selected CSV files to this directory

3. **Clean up original data**
   - This and all feature steps now happen exclusively in the extrapolation directory
   - For each project PK, find the latest date from the sequence in whose CSV file it first appears
   - For each project, extract category and labels from that CSV file
   - In all other files, override the project's category and labels there if it exists, else add it as new row 

4. **Extrapolation Phase Preparation**
   - Create a date sequence for the extrapolation region [x,y] with date intervals based on the specified timestep unit
     - The first date, x, is the earliest project creation date, the last date must be smaller than the first CSV file date, y
   - For each date in the extrapolation sequence:
     - Create a new CSV file with a stripped down structure of the original files
     - Columns: `name`, `homepage`, `y_property`, `category`, `labels`
     - Rows: Fill with unique project keys (name, homepage),
       - Fill project's category and label from the project-(category, labels) collection from the "previous step"


5. **Synthesis Phase**
   - Combine the original and extrapolated CSV files
   - The synthesized dataset can be used with the existing visualization module

### Visualization
- The existing visualization module can be used without modification
- A vertical dashed line will be added at the boundary between extrapolated and actual data (at date y)
- The x-axis will now span from the earliest project creation date to the latest CSV file date [x, z]

## Implementation Checklist

- [x] Set up project structure
- [x] Implement collection of unique project keys and creation dates
- [x] Create temporary workspace with selected CSV files
- [x] Clean up original data in selected CSV files
- [x] Implement date sequence generation for extrapolation region
- [x] Implement extrapolation of project data
- [ ] Implement synthesis of original and extrapolated data
- [ ] Add visual indicator for extrapolation boundary
- [x] Add comprehensive error handling and warnings
- [x] Create example usage script
- [ ] Add documentation
- [x] Test example usage script

## Current Progress

- [x] Initial analysis of data structure
- [x] Project requirements document
- [x] Implementation of collection phase
- [x] Implementation of workspace preparation
- [x] Implementation of data cleanup phase
- [x] Implementation of extrapolation phase
- [ ] Implementation of synthesis phase
- [x] Testing and validation of extrapolation phase

## Notes and Considerations

- The extrapolation approach assumes that projects maintain relatively stable metrics before their first appearance in the CSV files
- For some metrics like star count, a linear or exponential growth model might be more appropriate than a constant value
- The extrapolation directory is temporary by default, but can be preserved for debugging or further analysis
- Performance considerations for processing and generating many CSV files
- The synthesized dataset may be significantly larger than the original dataset, especially for fine-grained timestep units

## Lessons Learned

- Working with historical data requires careful handling of date formats and timezones
- Extrapolation should be clearly distinguished from actual data to avoid misinterpretation
- Creating a separate module for extrapolation keeps the codebase modular and maintainable

## Possible Optimizations and Future Improvements

- Implement different extrapolation models (constant, linear, exponential) based on the property type
- Add option to extrapolate only specific projects or labels
- Optimize memory usage when processing large datasets
- Add interactive visualization options to toggle between original and extrapolated views
- Implement caching of intermediate results to speed up repeated analyses
