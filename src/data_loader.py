"""
Data loading functions
Simple functions to load and check the dataset
"""

import pandas as pd
import numpy as np


def load_dataset(filepath):
    """Load CSV file"""
    print("Loading dataset...")
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded {len(data)} rows and {len(data.columns)} columns")
        return data
    except:
        print("Error loading file!")
        return None


def check_dataset(data):
    """Check the dataset for problems"""
    print("\nChecking dataset...")
    
    # Check for missing values
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print("Found missing values:")
        print(missing[missing > 0])
    else:
        print("No missing values found")
    
    # Check for duplicates
    duplicates = data.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Check class distribution if Class column exists
    if 'Class' in data.columns:
        normal = (data['Class'] == 0).sum()
        fraud = (data['Class'] == 1).sum()
        print(f"\nClass distribution:")
        print(f"Normal (0): {normal} ({normal/len(data)*100:.2f}%)")
        print(f"Fraud (1): {fraud} ({fraud/len(data)*100:.2f}%)")
        print(f"Imbalance ratio: {normal/fraud:.2f}:1")
    
    return data


def get_features_labels(data, exclude_cols=None):
    """Get features (X) and labels (y) from dataset"""
    if exclude_cols is None:
        exclude_cols = ['Class', 'Time']
    
    # Remove columns we don't want
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    X = data[feature_cols]
    y = data['Class']
    
    return X, y
