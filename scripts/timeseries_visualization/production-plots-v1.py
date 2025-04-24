#!/usr/bin/env python
"""
Example script demonstrating the use of the timeseries_visualization module.

This script shows how to create time series visualizations for different metrics
and label aggregations using the best-of-atomistic-machine-learning dataset.
"""

# %%
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml

# Add the src directory to the Python path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
    ),
)

from analysis.timeseries_visualization import (
    create_timeseries_visualization,
    TimeSeriesVisualizer,
)
# %%
# Create output directory
current_dir = Path.cwd()
dirpath = current_dir / "production-plot-v1"
dirpath.mkdir(exist_ok=True)

# %%
# categories and their titles in yaml format
cat_titles_yaml = """
- category: "active-learning"
  title: "Active learning"
- category: "community"
  title: "Community resources"
- category: "datasets"
  title: "Datasets"
- category: "data-structures"
  title: "Data Structures"
- category: "ml-dft"
  title: "Density functional theory (ML-DFT)"
- category: "educational"
  title: "Educational Resources"
- category: "xai"
  title: "Explainable Artificial intelligence (XAI)"
- category: "ml-esm"
  title: "Electronic structure methods (ML-ESM)"
- category: "general-tool"
  title: "General Tools"
- category: "generative"
  title: "Generative Models"
- category: "ml-iap"
  title: "Interatomic Potentials (ML-IAP)"
- category: "language-models"
  title: "Language Models"
- category: "materials-discovery"
  title: "Materials Discovery"
- category: "math"
  title: "Mathematical tools"
- category: "md"
  title: "Molecular Dynamics"
- category: "probabilistic"
  title: "Probabilistic ML"
- category: "reinforcement-learning"
  title: "Reinforcement Learning"
- category: "rep-eng"
  title: "Representation Engineering"
- category: "rep-learn"
  title: "Representation Learning"
- category: "uip"
  title: "Universal Potentials"
- category: "unsupervised"
  title: "Unsupervised Learning"
- category: "visualization"
  title: "Visualization"
- category: "ml-wft"
  title: "Wavefunction methods (ML-WFT)"
"""

# now as dictionary
cat_titles = {
    "active-learning": "Active learning",
    "community": "Community resources",
    "datasets": "Datasets",
    "data-structures": "Data Structures",
    "ml-dft": "Density functional theory (ML-DFT)",
    "educational": "Educational Resources",
    "xai": "Explainable Artificial intelligence (XAI)",
    "ml-esm": "Electronic structure methods (ML-ESM)",
    "general-tool": "General Tools",
    "generative": "Generative Models",
    "ml-iap": "Interatomic Potentials (ML-IAP)",
    "language-models": "Language Models",
    "materials-discovery": "Materials Discovery",
    "math": "Mathematical tools",
    "md": "Molecular Dynamics",
    "probabilistic": "Probabilistic ML",
    "reinforcement-learning": "Reinforcement Learning",
    "rep-eng": "Representation Engineering",
    "rep-learn": "Representation Learning",
    "uip": "Universal Potentials",
    "unsupervised": "Unsupervised Learning",
    "visualization": "Visualization",
    "ml-wft": "Wavefunction methods (ML-WFT)",
}

# %%
# Load BAML dataset
baml = pd.read_csv(f"../../analysis/history/2025-04-09_projects.csv", index_col=0)

# %%
# for column 'show', get absolute numbers of 'True' and 'False'
print(baml['show'].value_counts())
# True 268, False 238 = 53 %, 47 %

# %% 
# print the average and median value of the column 'created_at'
# Convert string dates to datetime objects and handle empty cells
baml['created_at_dt'] = pd.to_datetime(baml['created_at'], errors='coerce')

# Get current date for age calculations
current_date = pd.Timestamp('2025-04-14')

# Calculate date statistics
mean_date = baml['created_at_dt'].mean()
median_date = baml['created_at_dt'].median()
min_date = baml['created_at_dt'].min()
max_date = baml['created_at_dt'].max()

# Calculate ages (time deltas from current date)
mean_age = current_date - mean_date
median_age = current_date - median_date
min_age = current_date - min_date
max_age = current_date - max_date

# Print date values
print(f"Mean created_at: {mean_date}")
print(f"Median created_at: {median_date}")
print(f"Minimum created_at: {min_date}")
print(f"Maximum created_at: {max_date}")

# Function to convert timedelta to years, months, days
def timedelta_to_ymd(td):
    # Convert to days first
    days = td.days
    
    # Calculate years
    years = days // 365
    days = days % 365
    
    # Calculate months (approximate)
    months = days // 30
    days = days % 30
    
    return f"{years} years, {months} months, {days} days"

# Print age values in years, months, days
print(f"\nAges relative to current date ({current_date}):")
print(f"Mean age: {timedelta_to_ymd(mean_age)}")
print(f"Median age: {timedelta_to_ymd(median_age)}")
print(f"Age of oldest entry: {timedelta_to_ymd(min_age)}")
print(f"Age of newest entry: {timedelta_to_ymd(max_age)}")

# Count how many NaN values are in the created_at column
print(f"\nNumber of missing created_at values: {baml['created_at_dt'].isna().sum()}")

# %% 
# What percentage of projects have a 'single-paper' label?
print(f"\nPercentage of projects with 'single-paper' label: {baml['labels'].str.contains('single-paper').mean() * 100:.2f}%")

# %%
# extract dependent_project_count
colname = "dependent_project_count"

# convert dependent_project_count to int
baml[colname] = pd.to_numeric(baml[colname], downcast="integer")

# sort by dependent_project_count and show top 20
# show only name, dependent_project_count, category, and labels
# then save to csv file
baml_extract = baml.sort_values([colname], ascending=[False]).head(20)[
    ["name", colname, "category", "labels"]
]
# Add rank_by_dpc column (1, 2, 3, etc.)
baml_extract['rank_by_dpc'] = range(1, len(baml_extract) + 1)
baml_extract.to_csv(dirpath / f"{colname}.csv", index=False)
baml_extract

# %%
# Get median age of selection of projects from column `created_at`.
median_age = baml['created_at_dt'].median()
selected_project_names = ["PyG Models", "Deep Graph Library (DGL)", "RDKit", "DeepChem", "e3nn"]
selected_projects = baml[baml['name'].isin(selected_project_names)]
median_age = selected_projects['created_at_dt'].median()
print(f"\nMedian age of selection of projects: {timedelta_to_ymd(median_age)}")

# %%
# extract dependent_project_count
colname = "dependent_project_count"

# now again get an extract sorted by dependent_project_count,
# but include all projects and drop NaN values
baml_extract = baml.sort_values([colname], ascending=[False])[
    ["name", colname, "category", "labels"]
].dropna(subset=[colname])

# Function to check if 'single-paper' is in labels
def contains_single_paper(row):
    if isinstance(row['labels'], str):
        # Convert string representation of list to actual list if needed
        try:
            labels_list = eval(row['labels']) if row['labels'].startswith('[') else [row['labels']]
        except (SyntaxError, ValueError):
            labels_list = [row['labels']]
    elif isinstance(row['labels'], list):
        labels_list = row['labels']
    else:
        labels_list = []
    
    return any('single-paper' in label.lower() for label in labels_list if isinstance(label, str))

# Drop projects with 'single-paper' in labels
baml_extract = baml_extract[~baml_extract.apply(contains_single_paper, axis=1)]
print(f"After dropping NaN and single-paper projects: {len(baml_extract)} projects remaining")

# Add rank_by_dpc column (1, 2, 3, etc.)
baml_extract['rank_by_dpc'] = range(1, len(baml_extract) + 1)

# Filter for projects with specific strings in category or labels
target_strings = ['benchmarking', 'data-structures', 'math', 'workflows']

# Function to check if any target string is in category or labels
def contains_target_strings(row):
    # Check category (string)
    if isinstance(row['category'], str) and any(s in row['category'].lower() for s in target_strings):
        return True
    
    # Check labels (list of strings)
    if isinstance(row['labels'], str):
        # Convert string representation of list to actual list if needed
        try:
            labels_list = eval(row['labels']) if row['labels'].startswith('[') else [row['labels']]
        except (SyntaxError, ValueError):
            labels_list = [row['labels']]
    elif isinstance(row['labels'], list):
        labels_list = row['labels']
    else:
        labels_list = []
    
    # Check if any target string is in any label
    return any(any(s in label.lower() for s in target_strings) for label in labels_list if isinstance(label, str))

# Apply filter
filtered_baml_extract = baml_extract[baml_extract.apply(contains_target_strings, axis=1)]
print(f"Filtered from {len(baml_extract)} to {len(filtered_baml_extract)} projects")
filtered_baml_extract.to_csv(dirpath / f"dpc_all-projects_rse-labels-only.csv", index=False)
filtered_baml_extract



# %%
# extract star_count
colname = "star_count"

# convert star_count to int
baml[colname] = pd.to_numeric(baml[colname], downcast="integer")

# sort by star_count and show top 20
# show only name, star_count, category, and labels
# then save to csv file
baml_extract = baml.sort_values([colname], ascending=[False]).head(20)[
    ["name", colname, "category", "labels"]
]
baml_extract.to_csv(dirpath / f"{colname}.csv", index=False)
baml_extract

# %%
# Time series aggregation plot v1

# force matplotlib to use light background style
plt.style.use("default")
# plt.style.use("seaborn-whitegrid")
# plt.style.use("dark_background")

# Define aggregated labels
labels_agg1 = {
    "Interatomic Potentials": ["ml-iap", "uip"],
    "Representation Learning": ["rep-learn", "rep-eng"],
    "DFT & Electronic Structure": ["ml-dft", "ml-electronic-structure"],
    "Materials Discovery": ["materials-discovery", "structure-prediction"],
    "General Tools": ["general-tool", "cheminformatics"],
}

filename = "projectrank_agg1_ts1"
filepath = os.path.join(dirpath, filename)

df, fig = create_timeseries_visualization(
    y_property="projectrank",
    labels=labels_agg1,
    aggregation_method="sum",
    normalize=False,
    time_step=1,
    output_data_path=f"{filepath}.csv",
    output_plot_path=f"{filepath}.png",
    title="Evolution of ProjectRank by Category",
    ylabel="Total ProjectRank",
    interpolate_resource=True,
)

plt.show()


# %%
# Time series aggregation plot v2
# Define aggregated labels
labels_agg2 = {
    # "Community": ["community", "educational"],
    "Datasets": ["datasets", "community", "educational"],
    "Electronic Structure": ["ml-esm", "ml-dft", "ml-wft"],
    "Language Models": ["language-models"],
    "Generative Models": ["generative", "material-discovery", "probabilistic", "unsupervised"],
    # "Descriptors": ["rep-eng", "general-tools"],
    "Descriptors": ["rep-eng"],
    "Representation Learning": ["rep-learn", "data-structures", "math"],
    "Interatomic Potentials": ["ml-iap", "md"],
    "Universal Potentials": ["uip"],
    # "Active Learning": ["active-learning"],
    # "Utilities": ["general-tools", "data-structures", "math", "visualization", "xai", "reinforcement-learning"],
    # "Unsupervised Learning": ["unsupervised"],
}
stack_order = [
    # "Community",
    "Datasets",
    "Descriptors",
    "Representation Learning",
    "Electronic Structure",
    "Interatomic Potentials",
    "Universal Potentials",
    "Generative Models",
    "Language Models",
    # "Utilities",
    # "Active Learning",
    # "Unsupervised Learning",
]
legend_order = stack_order

filename = "agg2_projectrank_ts1"
# filename = "agg2_mdownloads_ts20"
filepath = os.path.join(dirpath, filename)

# force matplotlib to use light background style
plt.style.use("default")
# plt.style.use("seaborn-whitegrid")
# plt.style.use("dark_background")

df, fig = create_timeseries_visualization(
    # title="",
    plot_type="stacked",
    labels=labels_agg2,
    stack_order=stack_order,
    legend_order=legend_order,
    aggregation_method="sum",
    normalize=False,
    interpolate_resource=True,
    output_data_path=f"{filepath}.csv",
    output_plot_path=f"{filepath}.png", # .pdf
    fontsize=16,
    # projectrank # #
    y_property="projectrank",
    ylabel="Aggregated Project Rank",
    time_step=1,
    # # # monthly downloads # #
    # y_property="monthly_downloads",
    # ylabel="Monthly Downloads",
    # time_step=24,  # monthly downloads
)

plt.show()



# %%
# Example usage
df, fig = create_timeseries_visualization(
    y_property='projectrank',
    labels={'ML Libraries': ['pytorch', 'tensorflow'], 'Atomistic Tools': ['ase', 'pymatgen']},
    # Stack order (bottom to top)
    stack_order=['ML Libraries', 'Atomistic Tools'],
    # Legend order (can be different from stack order)
    legend_order=['Atomistic Tools', 'ML Libraries']
)
# %%
