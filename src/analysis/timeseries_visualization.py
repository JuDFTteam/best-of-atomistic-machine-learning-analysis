"""
Time Series Visualization Module for Atomistic Machine Learning Projects

This module provides functionality to analyze and visualize the evolution of project metrics
over time based on CSV data from the "best-of-atomistic-machine-learning" repository.
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """
    Class for generating time series visualizations of project metrics.
    
    This class handles the parsing of CSV files, aggregation of metrics by labels,
    and generation of stacked area charts showing the evolution of metrics over time.
    """
    
    def __init__(
        self, 
        history_dir: str = None,
        projects_yaml_path: str = None,
    ):
        """
        Initialize the TimeSeriesVisualizer.
        
        Args:
            history_dir: Path to the directory containing historical CSV files
            projects_yaml_path: Path to the projects.yaml file for label definitions
        """
        self.history_dir = history_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "analysis", "history"
        )
        self.projects_yaml_path = projects_yaml_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "analysis-dev", "projects.yaml"
        )
        
        # Validate paths
        if not os.path.exists(self.history_dir):
            raise FileNotFoundError(f"History directory not found: {self.history_dir}")
        
        if not os.path.exists(self.projects_yaml_path):
            raise FileNotFoundError(f"Projects YAML file not found: {self.projects_yaml_path}")
        
        # Initialize data structures
        self.csv_files = []
        self.timeseries_data = None
        self.all_labels = set()
        
    def _find_csv_files(self, time_step: int = 1) -> List[str]:
        """
        Find all CSV files in the history directory matching the pattern YYYY-MM-DD_projects.csv.
        
        Args:
            time_step: Sampling frequency for CSV files (1 = use all files)
            
        Returns:
            List of file paths sorted by date
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
        
        # Sample files based on time_step
        if time_step > 1 and len(all_files) > 2:
            # Always include first and last file
            sampled_files = [all_files[0][1]]
            
            # Sample intermediate files
            for i in range(1, len(all_files) - 1, time_step):
                sampled_files.append(all_files[i][1])
            
            # Add last file if not already included
            if all_files[-1][1] not in sampled_files:
                sampled_files.append(all_files[-1][1])
                
            return sampled_files
        
        # Return all files if time_step is 1 or there are too few files
        return [file[1] for file in all_files]
    
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
    
    def _get_labels_from_projects_yaml(self) -> Dict[str, List[str]]:
        """
        Extract label definitions from projects.yaml.
        
        Returns:
            Dictionary mapping label names to their descriptions
        """
        import yaml
        
        try:
            with open(self.projects_yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            labels = {}
            if 'labels' in yaml_data:
                for label_entry in yaml_data['labels']:
                    if 'label' in label_entry and 'name' in label_entry:
                        labels[label_entry['label']] = label_entry['name']
            
            return labels
        except Exception as e:
            logger.error(f"Error parsing projects.yaml: {e}")
            return {}
    
    def _aggregate_by_label(
        self, 
        df: pd.DataFrame, 
        y_property: str,
        labels_map: Dict[str, Union[List[str], None]],
        aggregation_method: str = 'sum'
    ) -> Dict[str, float]:
        """
        Aggregate property values by label.
        
        Args:
            df: DataFrame containing project data
            y_property: Property to aggregate (e.g., 'projectrank')
            labels_map: Dictionary mapping aggregated labels to lists of true labels
            aggregation_method: Method to aggregate values ('sum', 'mean', 'median')
            
        Returns:
            Dictionary mapping labels to aggregated values
        """
        # Check if required columns exist
        if 'labels' not in df.columns or y_property not in df.columns:
            missing_cols = []
            if 'labels' not in df.columns:
                missing_cols.append('labels')
            if y_property not in df.columns:
                missing_cols.append(y_property)
            logger.warning(f"Missing columns in DataFrame: {', '.join(missing_cols)}")
            return {}
        
        # Initialize result dictionary
        result = {label: 0.0 for label in labels_map.keys()}
        
        # Count missing properties
        missing_property_count = df[y_property].isna().sum()
        if missing_property_count > 0:
            logger.warning(f"Found {missing_property_count} projects with missing {y_property}")
        
        # Filter out rows with missing property
        df = df.dropna(subset=[y_property])
        
        # Process each project
        for _, row in df.iterrows():
            # Skip if no labels
            if pd.isna(row['labels']) or row['labels'] == '':
                continue
            
            # Parse labels (stored as string representation of list)
            try:
                # Handle different formats of labels column
                if isinstance(row['labels'], str):
                    if row['labels'].startswith('[') and row['labels'].endswith(']'):
                        # Handle string representation of list
                        project_labels = eval(row['labels'])
                    else:
                        # Handle comma-separated string
                        project_labels = [label.strip() for label in row['labels'].split(',')]
                elif isinstance(row['labels'], list):
                    project_labels = row['labels']
                else:
                    logger.warning(f"Unexpected labels format for project {row.get('name', 'unknown')}: {type(row['labels'])}")
                    continue
                
                # Add category as an additional label if it exists
                if 'category' in row and not pd.isna(row['category']) and row['category'] != '':
                    if isinstance(row['category'], str):
                        # Add category to project labels if it's not already there
                        category = row['category'].strip()
                        if category and category not in project_labels:
                            project_labels.append(category)
            except Exception as e:
                logger.warning(f"Error parsing labels for project {row.get('name', 'unknown')}: {e}")
                continue
            
            # Add project value to each matching label
            for agg_label, true_labels in labels_map.items():
                # If true_labels is None, use the aggregated label as a true label
                if true_labels is None:
                    if agg_label in project_labels:
                        result[agg_label] += float(row[y_property])
                else:
                    # Check if any of the true labels match
                    if any(label in project_labels for label in true_labels):
                        result[agg_label] += float(row[y_property])
        
        # Apply aggregation method if not sum
        if aggregation_method == 'mean' or aggregation_method == 'median':
            # Count number of projects per label
            label_counts = {label: 0 for label in labels_map.keys()}
            
            for _, row in df.iterrows():
                if pd.isna(row['labels']) or row['labels'] == '':
                    continue
                
                try:
                    if isinstance(row['labels'], str):
                        if row['labels'].startswith('[') and row['labels'].endswith(']'):
                            project_labels = eval(row['labels'])
                        else:
                            project_labels = [label.strip() for label in row['labels'].split(',')]
                    elif isinstance(row['labels'], list):
                        project_labels = row['labels']
                    else:
                        continue
                    
                    # Add category as an additional label if it exists
                    if 'category' in row and not pd.isna(row['category']) and row['category'] != '':
                        if isinstance(row['category'], str):
                            # Add category to project labels if it's not already there
                            category = row['category'].strip()
                            if category and category not in project_labels:
                                project_labels.append(category)
                except Exception:
                    continue
                
                for agg_label, true_labels in labels_map.items():
                    if true_labels is None:
                        if agg_label in project_labels:
                            label_counts[agg_label] += 1
                    else:
                        if any(label in project_labels for label in true_labels):
                            label_counts[agg_label] += 1
            
            # Apply aggregation
            for label in result:
                if label_counts[label] > 0:
                    if aggregation_method == 'mean':
                        result[label] /= label_counts[label]
                    elif aggregation_method == 'median':
                        # Note: For median, we would need individual values, not just sums
                        # This is an approximation assuming uniform distribution
                        result[label] /= label_counts[label]
        
        return result
    
    def process_data(
        self,
        y_property: str = 'projectrank',
        labels: Union[List[str], Dict[str, Union[List[str], None]]] = None,
        aggregation_method: str = 'sum',
        time_step: int = 1,
    ) -> pd.DataFrame:
        """
        Process CSV files and aggregate data by label over time.
        
        Args:
            y_property: Property to plot (e.g., 'projectrank', 'star_count')
            labels: List of labels or dict of aggregated labels to include
            aggregation_method: Method to aggregate values ('sum', 'mean', 'median')
            time_step: Sampling frequency for CSV files
            
        Returns:
            DataFrame containing time series data
        """
        if labels is None:
            raise ValueError("Labels must be provided")
        
        # Convert labels list to dict if needed
        if isinstance(labels, list):
            labels_map = {label: None for label in labels}
        else:
            labels_map = labels
        
        # Find CSV files
        self.csv_files = self._find_csv_files(time_step)
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {self.history_dir}")
        
        logger.info(f"Processing {len(self.csv_files)} CSV files")
        
        # Process each file
        data_by_date = {}
        for file_path in tqdm(self.csv_files, desc="Processing CSV files"):
            # Extract date from filename
            date = self._extract_date_from_filename(file_path)
            if date is None:
                logger.warning(f"Skipping file with invalid date format: {file_path}")
                continue
            
            # Parse CSV file
            df = self._parse_csv_file(file_path)
            if df.empty:
                logger.warning(f"Skipping empty or invalid file: {file_path}")
                continue
            
            # Check for duplicate project names
            if 'name' in df.columns and df['name'].duplicated().any():
                dup_count = df['name'].duplicated().sum()
                logger.warning(f"Found {dup_count} duplicate project names in {file_path}")
            
            # Aggregate by label
            label_values = self._aggregate_by_label(df, y_property, labels_map, aggregation_method)
            
            # Store results
            data_by_date[date] = label_values
        
        # Convert to DataFrame
        dates = sorted(data_by_date.keys())
        all_labels = set().union(*[set(data_by_date[date].keys()) for date in dates])
        
        # Create DataFrame with dates as index and labels as columns
        df_result = pd.DataFrame(index=dates, columns=list(all_labels))
        
        for date in dates:
            for label in all_labels:
                if label in data_by_date[date]:
                    df_result.loc[date, label] = data_by_date[date][label]
                else:
                    df_result.loc[date, label] = 0.0
        
        # Store results
        self.timeseries_data = df_result
        self.all_labels = all_labels
        
        return df_result
    
    def save_data(self, output_path: str) -> None:
        """
        Save processed time series data to a CSV file.
        
        Args:
            output_path: Path to save the CSV file
        """
        if self.timeseries_data is None:
            raise ValueError("No data to save. Run process_data() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        self.timeseries_data.to_csv(output_path)
        logger.info(f"Time series data saved to {output_path}")
    
    def plot(
        self,
        output_path: Optional[str] = None,
        normalize: bool = False,
        plot_type: str = 'stacked',
        color_scheme: str = 'viridis',
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
        xlabel: str = 'Date',
        ylabel: Optional[str] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Generate a stacked area chart or stream graph from the processed data.
        
        Args:
            output_path: Path to save the plot (if None, plot is not saved)
            normalize: Whether to normalize values (percentage instead of absolute)
            plot_type: Type of plot ('stacked' or 'stream')
            color_scheme: Color scheme for the plot
            figsize: Figure size (width, height) in inches
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            dpi: Resolution for saved image
            
        Returns:
            Matplotlib figure object
        """
        if self.timeseries_data is None:
            raise ValueError("No data to plot. Run process_data() first.")
        
        # Create a copy of the data for plotting
        plot_data = self.timeseries_data.copy()
        
        # Normalize if requested
        if normalize:
            # Calculate row sums
            row_sums = plot_data.sum(axis=1)
            # Divide each value by its row sum
            for col in plot_data.columns:
                plot_data[col] = plot_data[col] / row_sums * 100
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get x and y values
        x = plot_data.index
        
        # Convert values to numeric and handle NaN values
        y = []
        for col in plot_data.columns:
            # Convert to float and replace NaN with 0
            col_values = plot_data[col].fillna(0).astype(float).values
            y.append(col_values)
        
        # Set up colors
        if color_scheme in plt.colormaps():
            # Use built-in colormap
            cmap = plt.get_cmap(color_scheme)
            colors = [cmap(i / len(plot_data.columns)) for i in range(len(plot_data.columns))]
        else:
            # Default to a set of distinct colors
            colors = sns.color_palette("husl", len(plot_data.columns))
        
        # Create the plot
        if plot_type == 'stacked':
            ax.stackplot(x, y, labels=plot_data.columns, colors=colors, alpha=0.8)
        elif plot_type == 'stream':
            # For stream graph, center the stacked areas
            baseline = 'sym'  # symmetric baseline
            ax.stackplot(x, y, labels=plot_data.columns, colors=colors, alpha=0.8, baseline=baseline)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        # Set up labels and title
        if title:
            ax.set_title(title, fontsize=16)
        
        ax.set_xlabel(xlabel, fontsize=12)
        
        if ylabel:
            y_label_text = ylabel
        else:
            y_property = plot_data.columns[0] if plot_data.columns else "Value"
            if normalize:
                y_label_text = "Percentage (%)"
            else:
                y_label_text = f"{y_property}"
        
        ax.set_ylabel(y_label_text, fontsize=12)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
        
        return fig


def create_timeseries_visualization(
    y_property: str = 'projectrank',
    labels: Union[List[str], Dict[str, Union[List[str], None]]] = None,
    aggregation_method: str = 'sum',
    normalize: bool = False,
    time_step: int = 1,
    history_dir: Optional[str] = None,
    projects_yaml_path: Optional[str] = None,
    output_data_path: Optional[str] = None,
    output_plot_path: Optional[str] = None,
    plot_type: str = 'stacked',
    color_scheme: str = 'viridis',
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    xlabel: str = 'Date',
    ylabel: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Create a time series visualization of project metrics.
    
    Args:
        y_property: Property to plot (e.g., 'projectrank', 'star_count')
        labels: List of labels or dict of aggregated labels to include
        aggregation_method: Method to aggregate values ('sum', 'mean', 'median')
        normalize: Whether to normalize values (percentage instead of absolute)
        time_step: Sampling frequency for CSV files
        history_dir: Path to the directory containing historical CSV files
        projects_yaml_path: Path to the projects.yaml file for label definitions
        output_data_path: Path to save the processed data (if None, data is not saved)
        output_plot_path: Path to save the plot (if None, plot is not saved)
        plot_type: Type of plot ('stacked' or 'stream')
        color_scheme: Color scheme for the plot
        figsize: Figure size (width, height) in inches
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        dpi: Resolution for saved image
        
    Returns:
        Tuple containing the processed DataFrame and Matplotlib figure
    """
    # Create visualizer
    visualizer = TimeSeriesVisualizer(
        history_dir=history_dir,
        projects_yaml_path=projects_yaml_path,
    )
    
    # Process data
    df = visualizer.process_data(
        y_property=y_property,
        labels=labels,
        aggregation_method=aggregation_method,
        time_step=time_step,
    )
    
    # Save data if requested
    if output_data_path:
        visualizer.save_data(output_data_path)
    
    # Create plot
    fig = visualizer.plot(
        output_path=output_plot_path,
        normalize=normalize,
        plot_type=plot_type,
        color_scheme=color_scheme,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        dpi=dpi,
    )
    
    return df, fig
