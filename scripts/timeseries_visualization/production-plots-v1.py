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
# extract dependent_project_count
colname = "dependent_project_count"

# convert dependent_project_count to int
baml[colname] = pd.to_numeric(baml[colname], downcast="integer")

# sort by dependent_project_count and show top 20
# show only name, dependent_project_count, category, and labels
baml_extract = baml.sort_values([colname], ascending=[False]).head(20)[
    ["name", colname, "category", "labels"]
]
baml_extract.to_csv(dirpath / f"{colname}.csv", index=False)
baml_extract

# %%
# extract star_count
colname = "star_count"

# convert star_count to int
baml[colname] = pd.to_numeric(baml[colname], downcast="integer")

# sort by dependent_project_count and show top 20
# show only name, dependent_project_count, category, and labels
baml_extract = baml.sort_values([colname], ascending=[False]).head(20)[
    ["name", colname, "category", "labels"]
]
baml_extract.to_csv(dirpath / f"{colname}.csv", index=False)
baml_extract

# %%
# Time series aggregation plot 1
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
# Time series aggregation plot 2
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
    # projectrank # #
    y_property="projectrank",
    ylabel="Aggregated Project Rank",
    time_step=1,
    # # # monthly downloads # #
    # y_property="monthly_downloads",
    # ylabel="Monthly Downloads",
    # time_step=20,  # monthly downloads
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
