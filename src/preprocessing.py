"""
Preprocessing functions
Clean and prepare the data for modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def clean_data(data):
    """Remove duplicates and handle missing values"""
    print("Cleaning data...")
    
    # Remove duplicates
    before = len(data)
    data = data.drop_duplicates()
    after = len(data)
    print(f"Removed {before - after} duplicate rows")
    
    # Fill missing values with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")
    
    return data


def create_features(data):
    """Create new features"""
    print("Creating features...")
    
    # Log transform of Amount
    if 'Amount' in data.columns:
        data['Amount_log'] = np.log1p(data['Amount'])
        print("Created Amount_log feature")
    
    # Time features if Time column exists
    if 'Time' in data.columns:
        # Convert to hours
        data['Hour'] = (data['Time'] / 3600) % 24
        print("Created Hour feature")
    
    return data


def scale_features(X_train, X_test, X_val=None, method='standard'):
    """Scale features using StandardScaler or MinMaxScaler"""
    print(f"Scaling features using {method}...")
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Fit on training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    
    # Transform test data
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # Transform validation data if provided
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns
        )
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    return X_train_scaled, X_test_scaled, scaler


def split_train_test(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets"""
    print("Splitting data...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
