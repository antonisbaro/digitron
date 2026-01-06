"""
Main executable script for HMM-based Spoken Digit Classification (Steps 9-13).

This script orchestrates the entire HMM pipeline:
1.  Prepares the Free Spoken Digit Dataset (FSDD), specifying the
    data type (np.float32).
2.  Runs a hyperparameter grid search for the HMMs using the validation set.
3.  Evaluates the best-performing HMM model on the unseen test set.
4.  Generates and displays confusion matrices for detailed performance analysis.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.patrec.main_assignment.common.data import prepare_fsdd_data
from src.patrec.main_assignment.hmm.pipeline import (
    run_hmm_hyperparameter_tuning,
    evaluate_hmm_models,
)
from src.patrec.main_assignment.common.utils import plot_confusion_matrix


# --- 1. Configuration ---
OUTPUT_DIR = "outputs/hmm_results"
DATA_ROOT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_MFCC = 13

# --- 2. Data Preparation (Step 9) ---
print("--- [Step 9] Preparing FSDD Data for HMMs ---")
# Request np.float32 data type.
train_dic, val_dic, test_dic, unique_labels = prepare_fsdd_data(
    data_root=DATA_ROOT_DIR, n_mfcc=N_MFCC, data_type=np.float32
)

# --- 3. Hyperparameter Tuning (Steps 10 & 11) ---
best_hmm_models, best_params, results_df = run_hmm_hyperparameter_tuning(
    train_dic, val_dic, unique_labels
)

# --- 4. Display Tuning Results ---
print("\n--- Hyperparameter Tuning Results Summary ---")
if not results_df.empty:
    try:
        # Create a pivot table for a clear view of the results.
        results_pivot = results_df.pivot_table(
            index=["covariance_type", "n_states"],
            columns="n_mixtures",
            values="accuracy"
        )
        pd.options.display.float_format = '{:.4f}'.format
        print(results_pivot)
    except Exception as e:
        print(f"Could not create pivot table. Error: {e}")
        print("Displaying raw results DataFrame instead:\n", results_df)
else:
    print("No results to display from hyperparameter tuning.")

# --- 5. Final Evaluation (Steps 12 & 13) ---
# This block runs only if a model was successfully trained.
if best_hmm_models:
    print("\n--- [Step 12] Evaluating Best Model on the Unseen Test Set ---")
    pred_test, true_test = evaluate_hmm_models(best_hmm_models, test_dic, unique_labels)
    test_accuracy = accuracy_score(true_test, pred_test)
    print(f"\nFinal Test Set Accuracy: {test_accuracy:.4f}")

    print("\n--- [Step 13] Generating Confusion Matrices ---")
    pred_val, true_val = evaluate_hmm_models(best_hmm_models, val_dic, unique_labels)
    val_accuracy = best_params.get('accuracy', accuracy_score(true_val, pred_val))

    cm_validation = confusion_matrix(true_val, pred_val, labels=unique_labels)
    cm_test = confusion_matrix(true_test, pred_test, labels=unique_labels)
    class_names = [str(label) for label in unique_labels]

    plot_confusion_matrix(
        cm=cm_validation,
        class_names=class_names,
        title=f"HMM Confusion Matrix - Validation Set\nAccuracy: {val_accuracy:.4f}",
        output_path=os.path.join(OUTPUT_DIR, "hmm_cm_validation.png"),
    )
    plot_confusion_matrix(
        cm=cm_test,
        class_names=class_names,
        title=f"HMM Confusion Matrix - Test Set\nAccuracy: {test_accuracy:.4f}",
        output_path=os.path.join(OUTPUT_DIR, "hmm_cm_test.png"),
    )
else:
    print("\nSkipping final evaluation because no best model was found during tuning.")

print("\nHMM classification pipeline complete.")