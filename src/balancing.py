"""
Functions for handling class imbalance
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def apply_smote(X, y):
    """Apply SMOTE to balance the dataset"""
    print("Applying SMOTE...")
    print(f"Before: {len(X)} samples")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"After: {len(X_resampled)} samples")
    print(f"Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    # Convert back to DataFrame
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled)
    
    return X_resampled, y_resampled


def apply_undersampling(X, y):
    """Apply random undersampling"""
    print("Applying undersampling...")
    print(f"Before: {len(X)} samples")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    
    print(f"After: {len(X_resampled)} samples")
    print(f"Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    # Convert back to DataFrame
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled)
    
    return X_resampled, y_resampled


def get_normal_data(X, y):
    """Get only normal (non-fraud) samples"""
    normal_mask = (y == 0)
    X_normal = X[normal_mask]
    y_normal = y[normal_mask]
    
    print(f"Extracted {len(X_normal)} normal samples")
    
    return X_normal, y_normal
