#!/usr/bin/env python
"""
Example script demonstrating the use of the timeseries_visualization module.

This script shows how to create time series visualizations for different metrics
and label aggregations using the best-of-atomistic-machine-learning dataset.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from analysis.timeseries_visualization import create_timeseries_visualization, TimeSeriesVisualizer


def basic_example():
    """
    Basic example showing projectrank evolution for selected labels.
    """
    # Define labels to include
    labels = ["ml-iap", "uip", "rep-learn", "ml-dft", "materials-discovery"]
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    df, fig = create_timeseries_visualization(
        y_property="projectrank",
        labels=labels,
        aggregation_method="sum",
        normalize=False,
        time_step=10,  # Sample every x-th file to reduce processing time
        output_data_path=os.path.join(output_dir, "projectrank_by_label.csv"),
        output_plot_path=os.path.join(output_dir, "projectrank_by_label.png"),
        title="Evolution of ProjectRank by Label",
        ylabel="Total ProjectRank",
    )
    
    plt.show()
    
    return df, fig


def interpolate_resource_false_example():
    """
    Example showing how resource-True projects (websites) are handled without interpolation.

    This example will not estimate values for resource-True projects.
    """
    # Define labels to include
    labels = ["datasets", "ml-dft"]
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    df, fig = create_timeseries_visualization(
        y_property="projectrank",
        labels=labels,
        aggregation_method="sum",
        normalize=False,
        time_step=10,  # Sample every x-th file to reduce processing time
        output_data_path=os.path.join(output_dir, "projectrank_by_label_interpolate_resource_false.csv"),
        output_plot_path=os.path.join(output_dir, "projectrank_by_label_interpolate_resource_false.png"),
        title="Evolution of ProjectRank by Label, interpolate_resource=False",
        ylabel="Total ProjectRank",
        interpolate_resource=False,
    )
    
    plt.show()
    
    return df, fig

def interpolate_resource_true_example():
    """
    Example showing how to estimate values for resource-True projects (websites).

    This example will estimate values for resource-True projects by using the average
    of non-resource projects in the same label.

    For the chosen labels, this should make a visible difference (many website projects).
    """
    # Define labels to include
    labels = ["datasets", "ml-dft"]
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    df, fig = create_timeseries_visualization(
        y_property="projectrank",
        labels=labels,
        aggregation_method="sum",
        normalize=False,
        time_step=10,  # Sample every x-th file to reduce processing time
        output_data_path=os.path.join(output_dir, "projectrank_by_label_interpolate_resource_true.csv"),
        output_plot_path=os.path.join(output_dir, "projectrank_by_label_interpolate_resource_true.png"),
        title="Evolution of ProjectRank by Label, interpolate_resource=True",
        ylabel="Total ProjectRank",
        interpolate_resource=True,
    )
    
    plt.show()
    
    return df, fig


def normalized_example():
    """
    Example showing normalized (percentage) view of star count evolution.
    """
    # Define labels to include
    labels = ["ml-iap", "uip", "rep-learn", "ml-dft", "materials-discovery"]
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    df, fig = create_timeseries_visualization(
        y_property="star_count",
        labels=labels,
        aggregation_method="sum",
        normalize=True,  # Show as percentage
        time_step=10,
        output_data_path=os.path.join(output_dir, "star_count_by_label_normalized.csv"),
        output_plot_path=os.path.join(output_dir, "star_count_by_label_normalized.png"),
        title="Relative Evolution of Star Count by Label",
        ylabel="Percentage (%)",
        plot_type="stream",  # Use stream graph instead of stacked area
        color_scheme="plasma",
    )
    
    plt.show()
    
    return df, fig


def aggregated_labels_example():
    """
    Example showing how to use aggregated labels.
    """
    # Define aggregated labels
    aggregated_labels = {
        "Interatomic Potentials": ["ml-iap", "uip"],
        "Representation Learning": ["rep-learn", "rep-eng"],
        "DFT & Electronic Structure": ["ml-dft", "ml-electronic-structure"],
        "Materials Discovery": ["materials-discovery", "structure-prediction"],
        "General Tools": ["general-tool", "cheminformatics"]
    }
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    df, fig = create_timeseries_visualization(
        y_property="projectrank",
        labels=aggregated_labels,
        aggregation_method="sum",
        normalize=False,
        time_step=10,
        output_data_path=os.path.join(output_dir, "projectrank_by_aggregated_label.csv"),
        output_plot_path=os.path.join(output_dir, "projectrank_by_aggregated_label.png"),
        title="Evolution of ProjectRank by Category",
        ylabel="Total ProjectRank",
    )
    
    plt.show()
    
    return df, fig


def contributor_count_example():
    """
    Example showing evolution of contributor count.
    """
    # Define aggregated labels
    aggregated_labels = {
        "Interatomic Potentials": ["ml-iap", "uip"],
        "Representation Learning": ["rep-learn", "rep-eng"],
        "DFT & Electronic Structure": ["ml-dft", "ml-electronic-structure"],
        "Materials Discovery": ["materials-discovery", "structure-prediction"],
        "General Tools": ["general-tool", "cheminformatics"]
    }
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    df, fig = create_timeseries_visualization(
        y_property="contributor_count",
        labels=aggregated_labels,
        aggregation_method="sum",
        normalize=False,
        time_step=10,
        output_data_path=os.path.join(output_dir, "contributor_count_by_category.csv"),
        output_plot_path=os.path.join(output_dir, "contributor_count_by_category.png"),
        title="Evolution of Contributor Count by Category",
        ylabel="Total Contributors",
    )
    
    plt.show()
    
    return df, fig


if __name__ == "__main__":
    print("Running basic example...")
    basic_example()

    print("Running interpolate resource example...")
    interpolate_resource_false_example()
    interpolate_resource_true_example()
    
    print("Running normalized example...")
    normalized_example()
    
    print("Running aggregated labels example...")
    aggregated_labels_example()
    
    print("Running contributor count example...")
    contributor_count_example()
