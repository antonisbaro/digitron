"""
This module contains common utility functions used across the main assignment,
such as plotting functions for evaluation visualization. This promotes code
reusability (DRY principle).
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    output_path: str = None,
):
    """
    Renders and displays a visually appealing confusion matrix using seaborn.

    Args:
        cm (np.ndarray): The confusion matrix to be plotted, typically generated
                         by `sklearn.metrics.confusion_matrix`.
        class_names (List[str]): A list of strings representing the names of the
                                 classes, used for labeling the axes.
        title (str): The title to be displayed above the plot.
        output_path (str, optional): If provided, the plot is saved to this file path.
                                     Defaults to None, which only displays the plot.
    """
    # Create a figure and axes for the plot.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use seaborn's heatmap for a professional-looking plot.
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 12} # Increase font size for better readability
    )

    # Set plot titles and labels.
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    # Ensure layout is tight to prevent labels from being cut off.
    plt.tight_layout()

    # Save the figure to a file if an output path is provided.
    if output_path:
        # Create the directory if it doesn't exist.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"INFO: Confusion matrix saved to {output_path}")

    # Display the plot.
    plt.show()


def plot_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    title: str = "Training and Validation Loss",
    output_path: str = None,
):
    """
    Plots the training and validation loss curves over epochs.

    This visualization is essential for diagnosing model training, helping to
    identify issues like overfitting, underfitting, or convergence problems.

    Args:
        train_loss (List[float]): A list of training loss values, one per epoch.
        val_loss (List[float]): A list of validation loss values, one per epoch.
        title (str): The title for the plot.
        output_path (str, optional): If provided, the plot is saved to this file path.
                                     Defaults to None.
    """
    # Create a figure for the plot.
    plt.figure(figsize=(12, 7))

    # Plot the training and validation loss data.
    plt.plot(train_loss, label="Training Loss", color="royalblue", marker='o', linestyle='-')
    plt.plot(val_loss, label="Validation Loss", color="darkorange", marker='o', linestyle='-')

    # Set plot titles and labels for clarity.
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the figure to a file if an output path is provided.
    if output_path:
        # Create the directory if it doesn't exist.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"INFO: Loss curves plot saved to {output_path}")

    # Display the plot.
    plt.show()