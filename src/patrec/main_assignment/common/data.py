"""
This module handles all data loading, parsing, and preparation for the main
assignment (Steps 9-14), specifically for the Free Spoken Digit Dataset (FSDD).
It provides a single entry point `prepare_fsdd_data` to orchestrate the process.
"""

import os
import subprocess
from glob import glob
from typing import List, Tuple, Callable, Dict, Any

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def download_fsdd_if_needed(data_root_dir: str) -> str:
    """
    Checks for the FSDD dataset directory and downloads it via git clone if not present.

    Args:
        data_root_dir (str): The root directory where the dataset folder should exist.

    Returns:
        str: The full path to the FSDD directory.

    Raises:
        RuntimeError: If git is not installed or the repository cannot be cloned.
    """
    fsdd_path = os.path.join(data_root_dir, "free-spoken-digit-dataset")
    if not os.path.exists(fsdd_path):
        print("FSDD dataset not found. Cloning from GitHub...")
        repo_url = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
        try:
            # Use subprocess to run the git clone command.
            # check=True will raise an exception if the command fails.
            subprocess.run(
                ["git", "clone", repo_url, fsdd_path],
                check=True,
                capture_output=True,
                text=True  # Capture output as text for better error messages
            )
            print("Dataset downloaded successfully.")
        except FileNotFoundError:
            # This error occurs if the 'git' command itself is not found.
            raise RuntimeError(
                "Git is not installed or not in the system's PATH. "
                "Please install Git to download the dataset automatically, "
                f"or manually download it from {repo_url}"
            )
        except subprocess.CalledProcessError as e:
            # This error occurs if git returns a non-zero exit code (e.g., network error).
            raise RuntimeError(
                f"Failed to clone FSDD repository. Git command failed with error: {e.stderr}"
            )
    return fsdd_path


def prepare_fsdd_data(data_root: str, n_mfcc: int = 13, data_type: type = np.float32) -> Tuple[Dict, Dict, Dict, List[int]]:
    """
    Orchestrates the entire data preparation pipeline for the FSDD dataset.

    This function performs the following steps:
    1.  Ensures the dataset is downloaded.
    2.  Parses filenames to extract labels and speaker information.
    3.  Extracts MFCC features from all audio files.
    4.  Splits data into a primary train/test set (e.g., 80/20 split).
    5.  Further splits the primary training set into a final train/validation set (80/20).
    6.  Computes scaling parameters (mean/std) from the final training set.
    7.  Applies scaling to the train, validation, and test sets.
    8.  Groups the processed data into dictionaries keyed by digit for easy access.

    Args:
        data_root (str): The path to the main 'data' directory of the project.
        n_mfcc (int): The number of Mel-frequency cepstral coefficients to extract.
        data_type (type): The target NumPy data type for the features.
                          Use np.float64 for HMMs (pomegranate) and
                          np.float32 for NNs (PyTorch).

    Returns:
        Tuple[Dict, Dict, Dict, List[int]]: A tuple containing:
            - train_dic: Data for the training set, grouped by digit.
            - val_dic: Data for the validation set, grouped by digit.
            - test_dic: Data for the test set, grouped by digit.
            - unique_labels: A sorted list of the unique digit labels (0-9).
    """
    # --- 1. Ensure dataset exists ---
    fsdd_path = download_fsdd_if_needed(data_root)
    recordings_path = os.path.join(fsdd_path, "recordings")

    # --- 2. Parse filenames ---
    print("Parsing FSDD filenames...")
    file_paths = sorted(glob(os.path.join(recordings_path, "*.wav")))
    if not file_paths:
        raise FileNotFoundError(f"No .wav files found in {recordings_path}")
    
    # Extract metadata by splitting "0_jackson_0.wav" into [digit, speaker, index]
    fnames = [os.path.basename(f).split(".")[0].split("_") for f in file_paths]
    labels = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]

    # --- 3. Extract MFCC Features ---
    window_sr = 8000  # The dataset's native sampling rate
    window_size = int(0.030 * window_sr)  # 30ms window
    hop_length = int(0.015 * window_sr)  # 15ms hop (50% overlap)

    features = []
    for f in tqdm(file_paths, desc="Extracting MFCC features"):
        wav, _ = librosa.load(f, sr=window_sr)
        mfccs = librosa.feature.mfcc(
            y=wav, sr=window_sr, n_mfcc=n_mfcc, n_fft=window_size, hop_length=hop_length
        ).T
        features.append(mfccs.astype(data_type)) # Set the data type here

    # --- 4. Split data into initial train/test sets (80/20) ---
    X_train_full, X_test_raw, y_train_full, y_test, spk_train_full, spk_test = train_test_split(
        features, labels, speakers, test_size=0.20, random_state=42, stratify=labels
    )
    
    # --- 5. Split train_full into final train/validation sets (80/20) ---
    X_train_raw, X_val_raw, y_train, y_val, spk_train, spk_val = train_test_split(
        X_train_full, y_train_full, spk_train_full, test_size=0.20, random_state=42, stratify=y_train_full
    )

    # --- 6. Fit scaler on the training data ONLY ---
    print("Fitting StandardScaler on the training data...")
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train_raw))

    # --- 7. Apply scaling to all sets ---
    X_train = [scaler.transform(x) for x in X_train_raw]
    X_val = [scaler.transform(x) for x in X_val_raw]
    X_test = [scaler.transform(x) for x in X_test_raw]

    # --- 8. Group data into dictionaries ---
    def _gather_in_dict(X: List[np.ndarray], y: List[int], spk: List[int]) -> Dict:
        """Helper to group data by digit."""
        data_dict = {}
        unique_y = sorted(list(set(y)))
        for digit in unique_y:
            # Find indices for the current digit
            indices = [i for i, label in enumerate(y) if label == digit]
            # Collect all samples for this digit
            data_dict[digit] = {
                "features": [X[i] for i in indices],
                "labels": [y[i] for i in indices],
                "speakers": [spk[i] for i in indices],
            }
        return data_dict

    train_dic = _gather_in_dict(X_train, y_train, spk_train)
    val_dic = _gather_in_dict(X_val, y_val, spk_val)
    test_dic = _gather_in_dict(X_test, y_test, spk_test)

    unique_labels = sorted(list(set(labels)))
    
    print("FSDD data preparation complete.")
    print(f"Total samples: {len(features)} -> Train: {len(y_train)}, Validation: {len(y_val)}, Test: {len(y_test)}")
    
    return train_dic, val_dic, test_dic, unique_labels