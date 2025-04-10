# Timeseries Visualization for Atomistic Machine Learning Projects

## Project Overview

This document outlines the requirements and design for creating a time series visualization of atomistic machine learning projects based on CSV data from the "best-of-atomistic-machine-learning" repository.

### Data Source
- CSV files in `analysis/history` following the pattern `YYYY-MM-DD_projects.csv`
- Files span from 2023-06-09 to 2025-04-09 (approximately 100 files)
- Each file contains metadata about atomistic machine learning projects
- Newer files may have additional columns compared to older files
- Project categories and labels are defined in `analysis-dev/projects.yaml`

### Visualization Goal
Create a stacked area chart or streamchart that shows the evolution of project metrics over time, categorized by project labels/categories for publication in an academic paper about atomistic machine learning tools.

## Functional Requirements

### Core Functionality
1. Parse CSV files from `analysis/history` directory
2. Extract time information from filenames
3. Aggregate project metrics by label/category
4. Generate stacked area charts showing the evolution of metrics over time

### Input Parameters
- **y_property**: Dynamic property to plot (default: `projectrank`)
- **labels**: List of labels or dict of aggregated labels to include (required, no default)
- **aggregation_method**: Method to aggregate property values (default: `sum`, options: `sum`, `mean`, `median`)
- **normalize**: Whether to normalize values (default: `False`)
- **time_step**: Sampling frequency for CSV files (default: `1` - use all files)

### Label Aggregation
- Support for aggregated labels where multiple true labels are combined
- Format: `{aggregated_label_name: [list_of_true_labels]}`
- If value is `None`, assume the key is a true label

### Data Handling
- Primary key for projects: `name` column
- Missing properties: Exclude from aggregation, print warning with count
- Missing dates: Skip and continue
- Duplicate project names: Print warning

## Technical Design

### Processing Pipeline
1. Scan directory for all CSV files matching the pattern
2. Sample files based on time_step parameter (always include first and last)
3. For each file:
   - Parse CSV data
   - Extract date from filename
   - Filter projects with the selected property
   - Aggregate property values by (true) label
   - For aggregated labels, aggregate the value of their true labels
4. Combine data into a time series
5. Save the timeseries data as a Pandas dataframe and save it as a new CSV file
6. Generate visualization

### Visualization Options
- Stacked area chart (default)
- Stream graph option
- Customizable color schemes
- Option for normalized view (percentage instead of absolute values)

## Implementation Checklist

- [x] Set up project structure
- [x] Implement CSV file parsing
- [x] Implement label aggregation logic
- [x] Implement property aggregation
- [x] Create basic stacked area chart
- [x] Add normalization option
- [x] Add color scheme options
- [x] Add stream graph alternative
- [x] Add comprehensive error handling and warnings
- [x] Create example usage script
- [x] Add documentation
- [x] Test example usage script
- [ ] Add parsing of category column and add it to labels

## Current Progress

- [x] Initial analysis of data structure
- [x] Project requirements document
- [x] Implementation completed
- [x] Basic examples working
- [ ] Add option to include projects with missing properties by interpolation (ask for clarification what this means)
- [ ] Add option to smooth the timeseries data (ask for clarification what this means)
- [ ] Advanced features and optimizations

## Notes and Considerations

- Older CSV files may have fewer columns than newer ones
- The number of projects grows over time as new ones are added
- Visual clarity is important for publication-quality figures
- Performance considerations for processing ~100 CSV files
- Some categories or labels found in CSV files might conflict with the ones in
  `analysis-dev/projects.yaml`. This can be if they have been renamed or
  removed.

## Lessons Learned

- Data type handling is critical when working with time series data in matplotlib
- Proper handling of NaN values and type conversion is necessary to avoid plotting errors
- Error handling is essential for robustness when processing multiple historical files

## Possible Optimizations and Future Improvements

- If processing is slow, a possible preprocessing step is one-hot encoding
  category and labels in a folder copy of the original CSV files. In the
  original CSV files, the labels are stored in one column as list of strings.
  This is inefficient for lookup.
- Add caching mechanism to avoid reprocessing all CSV files when only visualization parameters change
- Implement interactive visualization options (e.g., using Plotly or Bokeh)
- Add trend analysis features to highlight significant changes in metrics over time
- Improve handling of date ranges with missing data points
- Add export options for different file formats (SVG, PDF, etc.) for publication
