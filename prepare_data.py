"""
Helper script to download and prepare sample credit card fraud dataset.
This script helps set up a sample dataset if you don't have one ready.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import os


def download_creditcard_dataset(url: str = None, output_dir: str = 'data'):
    """
    Download credit card fraud dataset from Kaggle or use sample data.
    
    Note: For the actual Kaggle dataset, you'll need:
    1. Kaggle API credentials (kaggle.json)
    2. Install kaggle: pip install kaggle
    3. Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("CREDIT CARD FRAUD DETECTION - DATA PREPARATION")
    print("=" * 80)
    
    print("\nTo use this project, you need a credit card fraud dataset.")
    print("Recommended dataset: Credit Card Fraud Detection from Kaggle")
    print("URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("\nThe dataset should have:")
    print("- Features: V1, V2, ..., V28 (PCA-transformed features)")
    print("- Amount: Transaction amount")
    print("- Time: Seconds elapsed between transactions")
    print("- Class: Target variable (0 = Normal, 1 = Fraud)")
    
    print("\n" + "=" * 80)
    print("OPTION 1: Download from Kaggle (Recommended)")
    print("=" * 80)
    print("""
1. Install Kaggle API:
   pip install kaggle

2. Set up Kaggle credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save kaggle.json to ~/.kaggle/

3. Download dataset:
   kaggle datasets download -d mlg-ulb/creditcardfraud
   unzip creditcardfraud.zip -d data/
    """)
    
    print("\n" + "=" * 80)
    print("OPTION 2: Create Sample Dataset (For Testing)")
    print("=" * 80)
    
    # Automatically create sample dataset
    print("\nCreating sample dataset automatically...")
    create_sample_dataset(output_path / 'creditcard_sample.csv')
    print(f"\nSample dataset created at: {output_path / 'creditcard_sample.csv'}")
    print("Note: This is synthetic data for testing purposes only.")


def create_sample_dataset(filepath: Path, n_samples: int = 10000, fraud_ratio: float = 0.0017):
    """
    Create a synthetic sample dataset for testing.
    
    Args:
        filepath: Path to save the dataset
        n_samples: Number of samples to generate
        fraud_ratio: Ratio of fraud cases
    """
    print(f"\nCreating sample dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Generate features (V1-V28 are PCA components, so we'll use random normal)
    n_features = 28
    features = {}
    
    for i in range(1, n_features + 1):
        features[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Generate Amount (log-normal distribution)
    features['Amount'] = np.random.lognormal(3, 1.5, n_samples)
    features['Amount'] = np.clip(features['Amount'], 0, 50000)
    
    # Generate Time (sequential with some randomness)
    features['Time'] = np.arange(n_samples) * np.random.uniform(0.5, 2.0, n_samples)
    
    # Generate Class (imbalanced)
    n_fraud = int(n_samples * fraud_ratio)
    classes = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    classes[fraud_indices] = 1
    
    # Make fraud cases slightly different (anomalies)
    for idx in fraud_indices:
        # Add some anomaly to features
        anomaly_features = np.random.choice(n_features, size=5, replace=False)
        for feat_idx in anomaly_features:
            features[f'V{feat_idx + 1}'][idx] += np.random.normal(0, 3)
        features['Amount'][idx] *= np.random.uniform(1.5, 5.0)
    
    features['Class'] = classes
    
    # Create DataFrame
    df = pd.DataFrame(features)
    
    # Reorder columns
    cols = [f'V{i}' for i in range(1, n_features + 1)] + ['Amount', 'Time', 'Class']
    df = df[cols]
    
    # Save
    df.to_csv(filepath, index=False)
    
    print(f"Dataset created: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean() * 100:.2f}%)")
    print(f"Normal cases: {(df['Class'] == 0).sum()} ({(df['Class'] == 0).mean() * 100:.2f}%)")


if __name__ == "__main__":
    download_creditcard_dataset()

