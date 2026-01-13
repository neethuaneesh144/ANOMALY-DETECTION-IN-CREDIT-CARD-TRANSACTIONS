"""
Credit Card Fraud Detection Project
Main script to run all models and generate results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Import sklearn stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

# TensorFlow for autoencoder
TENSORFLOW_AVAILABLE = False
keras = None
layers = None
try:
    import tensorflow as tf
    # Try different import methods for different TensorFlow versions
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except:
        try:
            import keras
            from keras import layers
        except:
            keras = tf.keras
            layers = tf.keras.layers
    
    if keras is not None:
        tf.random.set_seed(42)
        TENSORFLOW_AVAILABLE = True
except Exception as e:
    print(f"Warning: TensorFlow/Keras not available. Autoencoder will not work. ({e})")

np.random.seed(42)

# Create folders if they don't exist
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('reports'):
    os.makedirs('reports')
if not os.path.exists('reports/figures'):
    os.makedirs('reports/figures')
if not os.path.exists('output'):
    os.makedirs('output')


def load_data(filepath):
    """Load the CSV file"""
    print("Loading data...")
    data = pd.read_csv(filepath)
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def check_data(data):
    """Check for missing values and class distribution"""
    print("\n=== Data Check ===")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Duplicates: {data.duplicated().sum()}")
    
    # Check class distribution
    if 'Class' in data.columns:
        fraud_count = (data['Class'] == 1).sum()
        normal_count = (data['Class'] == 0).sum()
        print(f"Normal transactions: {normal_count}")
        print(f"Fraud transactions: {fraud_count}")
        print(f"Fraud percentage: {(fraud_count/len(data))*100:.2f}%")
    return data


def preprocess_data(data):
    """Clean and preprocess the data"""
    print("\n=== Preprocessing ===")
    
    # Remove duplicates
    data = data.drop_duplicates()
    print(f"After removing duplicates: {data.shape[0]} rows")
    
    # Handle missing values (fill with median)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)
    
    # Feature engineering - add log of Amount if it exists
    if 'Amount' in data.columns:
        data['Amount_log'] = np.log1p(data['Amount'])
        print("Created Amount_log feature")
    
    # Separate features and target
    if 'Time' in data.columns:
        X = data.drop(['Time', 'Class'], axis=1)
    else:
        X = data.drop(['Class'], axis=1)
    y = data['Class']
    
    return X, y


def split_data(X, y):
    """Split into train, validation, and test sets"""
    print("\n=== Splitting Data ===")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def balance_data(X_train, y_train, method='smote'):
    """Balance the training data using SMOTE or undersampling"""
    print(f"\n=== Balancing Data ({method}) ===")
    
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_balanced.shape[0]} samples")
    elif method == 'undersample':
        undersampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)
        print(f"After undersampling: {X_balanced.shape[0]} samples")
    else:
        X_balanced, y_balanced = X_train, y_train
    
    return X_balanced, y_balanced


def train_supervised_models(X_train, y_train, X_test, y_test, balance_method='none'):
    """Train supervised learning models"""
    print(f"\n=== Training Supervised Models ({balance_method}) ===")
    
    results = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced' if balance_method == 'none' else None)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = {
        'model': lr,
        'predictions': lr_pred,
        'probabilities': lr_proba
    }
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced' if balance_method == 'none' else None)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'model': rf,
        'predictions': rf_pred,
        'probabilities': rf_proba
    }
    
    # SVM (only if not undersampled - takes too long)
    if balance_method != 'undersample':
        print("Training SVM...")
        svm = SVC(random_state=42, probability=True, class_weight='balanced' if balance_method == 'none' else None)
        # Use sample for speed
        if len(X_train) > 10000:
            sample_idx = np.random.choice(len(X_train), 10000, replace=False)
            svm.fit(X_train.iloc[sample_idx], y_train.iloc[sample_idx])
        else:
            svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_proba = svm.predict_proba(X_test)[:, 1]
        
        results['SVM'] = {
            'model': svm,
            'predictions': svm_pred,
            'probabilities': svm_proba
        }
    
    return results


def train_unsupervised_models(X_train_normal, X_test, y_test, contamination=0.01):
    """Train unsupervised anomaly detection models"""
    print(f"\n=== Training Unsupervised Models (contamination={contamination}) ===")
    
    results = {}
    
    # Isolation Forest
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X_train_normal)
    iso_pred = iso_forest.predict(X_test)
    iso_scores = iso_forest.score_samples(X_test)
    # Convert -1 (anomaly) to 1 (fraud), 1 (normal) to 0
    iso_pred_binary = (iso_pred == -1).astype(int)
    iso_scores = -iso_scores  # Invert so higher = more anomalous
    
    results['Isolation Forest'] = {
        'model': iso_forest,
        'predictions': iso_pred_binary,
        'scores': iso_scores
    }
    
    # One-Class SVM
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(nu=contamination)
    # Use sample for speed
    if len(X_train_normal) > 10000:
        sample_idx = np.random.choice(len(X_train_normal), 10000, replace=False)
        ocsvm.fit(X_train_normal.iloc[sample_idx])
    else:
        ocsvm.fit(X_train_normal)
    ocsvm_pred = ocsvm.predict(X_test)
    ocsvm_scores = ocsvm.score_samples(X_test)
    ocsvm_pred_binary = (ocsvm_pred == -1).astype(int)
    ocsvm_scores = -ocsvm_scores
    
    results['One-Class SVM'] = {
        'model': ocsvm,
        'predictions': ocsvm_pred_binary,
        'scores': ocsvm_scores
    }
    
    # LOF
    print("Training LOF...")
    lof = LocalOutlierFactor(contamination=contamination, novelty=True)
    lof.fit(X_train_normal)
    lof_pred = lof.predict(X_test)
    lof_scores = -lof.negative_outlier_factor_
    lof_pred_binary = (lof_pred == -1).astype(int)
    
    results['LOF'] = {
        'model': lof,
        'predictions': lof_pred_binary,
        'scores': lof_scores
    }
    
    return results


def train_autoencoder(X_train_normal, X_val_normal, X_test, input_dim):
    """Train autoencoder for anomaly detection"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras is not available. Cannot train autoencoder.")
    
    print("\n=== Training Autoencoder ===")
    
    # Build model
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("Training autoencoder...")
    history = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=50,
        batch_size=32,
        validation_data=(X_val_normal, X_val_normal) if X_val_normal is not None else None,
        verbose=1
    )
    
    # Calculate reconstruction error
    X_pred = autoencoder.predict(X_test, verbose=0)
    reconstruction_errors = np.mean(np.square(X_test - X_pred), axis=1)
    
    # Set threshold (99th percentile of normal data)
    X_train_pred = autoencoder.predict(X_train_normal, verbose=0)
    train_errors = np.mean(np.square(X_train_normal - X_train_pred), axis=1)
    threshold = np.percentile(train_errors, 99)
    
    # Predict anomalies
    ae_pred = (reconstruction_errors >= threshold).astype(int)
    
    return {
        'model': autoencoder,
        'predictions': ae_pred,
        'scores': reconstruction_errors,
        'threshold': threshold,
        'history': history.history
    }


def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate all evaluation metrics"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # False positive and false negative rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Balanced accuracy
    balanced_acc = (recall + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'fnr': fnr,
        'balanced_accuracy': balanced_acc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'confusion_matrix': cm
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    
    return metrics


def plot_confusion_matrix(cm, model_name, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(results_dict, y_test, save_path):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    for name, result in results_dict.items():
        if 'probabilities' in result:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            auc = roc_auc_score(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curves(results_dict, y_test, save_path):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 8))
    for name, result in results_dict.items():
        if 'probabilities' in result:
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            pr_auc = average_precision_score(y_test, result['probabilities'])
            plt.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_class_imbalance(data, save_path):
    """Plot class imbalance bar chart"""
    if 'Class' in data.columns:
        class_counts = data['Class'].value_counts()
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Normal (0)', 'Fraud (1)'], [class_counts[0], class_counts[1]], 
                       color=['blue', 'red'], alpha=0.7)
        plt.ylabel('Number of Transactions')
        plt.title('Class Distribution - Imbalance Visualization')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(data)*100:.2f}%)',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved class imbalance plot to {save_path}")


def plot_amount_distribution(data, save_path):
    """Plot transaction amount distribution for normal vs fraud"""
    if 'Amount' in data.columns and 'Class' in data.columns:
        plt.figure(figsize=(12, 6))
        
        normal_amounts = data[data['Class'] == 0]['Amount']
        fraud_amounts = data[data['Class'] == 1]['Amount']
        
        plt.hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
        
        plt.xlabel('Transaction Amount')
        plt.ylabel('Density')
        plt.title('Transaction Amount Distribution: Normal vs Fraud')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved amount distribution plot to {save_path}")
        
        # Also create log-scale version
        plt.figure(figsize=(12, 6))
        plt.hist(np.log1p(normal_amounts), bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(np.log1p(fraud_amounts), bins=50, alpha=0.7, label='Fraud', color='red', density=True)
        plt.xlabel('Log(Transaction Amount + 1)')
        plt.ylabel('Density')
        plt.title('Transaction Amount Distribution (Log Scale): Normal vs Fraud')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_log.png'), dpi=300)
        plt.close()
        print(f"Saved log-scale amount distribution plot to {save_path.replace('.png', '_log.png')}")


def plot_correlation_heatmap(X, save_path, max_features=30):
    """Plot correlation heatmap for features"""
    # Limit features if too many
    if X.shape[1] > max_features:
        # Select first max_features columns
        X_plot = X.iloc[:, :max_features]
    else:
        X_plot = X
    
    plt.figure(figsize=(12, 10))
    corr_matrix = X_plot.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Feature Correlation Heatmap (showing first {X_plot.shape[1]} features)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")


def plot_imbalance_comparison(metrics_list, save_path):
    """Plot comparison of balancing strategies for supervised models"""
    # Filter supervised models
    supervised_models = {}
    for m in metrics_list:
        model_name = m['model_name']
        if 'Logistic Regression' in model_name or 'Random Forest' in model_name or 'SVM' in model_name:
            # Extract base model name
            base_name = model_name.split('(')[0].strip()
            if base_name not in supervised_models:
                supervised_models[base_name] = {'No Balance': None, 'SMOTE': None, 'Undersample': None}
            
            if 'No Balance' in model_name:
                supervised_models[base_name]['No Balance'] = m
            elif 'SMOTE' in model_name:
                supervised_models[base_name]['SMOTE'] = m
            elif 'Undersample' in model_name:
                supervised_models[base_name]['Undersample'] = m
    
    if not supervised_models:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['precision', 'recall', 'f1_score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x_pos = np.arange(len(supervised_models))
        width = 0.25
        
        for i, (model_name, strategies) in enumerate(supervised_models.items()):
            values = []
            if strategies['No Balance']:
                values.append(strategies['No Balance'].get(metric, 0))
            else:
                values.append(0)
            
            if strategies['SMOTE']:
                values.append(strategies['SMOTE'].get(metric, 0))
            else:
                values.append(0)
            
            if strategies['Undersample']:
                values.append(strategies['Undersample'].get(metric, 0))
            else:
                values.append(0)
            
            ax.bar(x_pos[i] + width * (np.arange(len(values)) - 1), values, width, label=model_name if idx == 0 else "")
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(supervised_models.keys()), rotation=45, ha='right')
        if idx == 0:
            ax.legend(['No Balance', 'SMOTE', 'Undersample'])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved imbalance comparison plot to {save_path}")


def plot_precision_recall_vs_threshold(y_true, y_scores, model_name, save_path):
    """Plot Precision and Recall vs Threshold"""
    # Ensure y_true and y_scores have same length
    min_len = min(len(y_true), len(y_scores))
    y_true = y_true[:min_len]
    y_scores = y_scores[:min_len]
    
    # Normalize scores to 0-1 range for thresholding
    if y_scores.min() < 0 or y_scores.max() > 1:
        y_scores_normalized = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
    else:
        y_scores_normalized = y_scores
    
    thresholds = np.linspace(y_scores_normalized.min(), y_scores_normalized.max(), 100)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = (y_scores_normalized >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision and Recall vs Threshold - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved precision/recall vs threshold plot to {save_path}")


def main():
    """Main function to run everything"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 80)
    print("CREDIT CARD FRAUD DETECTION PROJECT")
    print("=" * 80)
    
    # Step 1: Load data
    data = load_data(args.data)
    data = check_data(data)
    
    # Step 1.5: Create EDA plots
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    plot_class_imbalance(data, "reports/figures/class_imbalance.png")
    if 'Amount' in data.columns:
        plot_amount_distribution(data, "reports/figures/amount_distribution.png")
    
    # Step 2: Preprocess
    X, y = preprocess_data(data)
    
    # Plot correlation heatmap (before scaling)
    if X.shape[1] <= 30:
        plot_correlation_heatmap(X, "reports/figures/correlation_heatmap.png")
    
    # Step 3: Split data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_data(X, y)
    
    # Step 4: Train supervised models with different balancing
    all_results = {}
    
    # No balancing
    print("\n" + "=" * 80)
    print("SUPERVISED MODELS - NO BALANCING")
    print("=" * 80)
    results_no_balance = train_supervised_models(X_train, y_train, X_test, y_test, 'none')
    all_results.update({f"{k} (No Balance)": v for k, v in results_no_balance.items()})
    
    # SMOTE
    print("\n" + "=" * 80)
    print("SUPERVISED MODELS - SMOTE")
    print("=" * 80)
    X_train_smote, y_train_smote = balance_data(X_train, y_train, 'smote')
    results_smote = train_supervised_models(X_train_smote, y_train_smote, X_test, y_test, 'smote')
    all_results.update({f"{k} (SMOTE)": v for k, v in results_smote.items()})
    
    # Undersampling
    print("\n" + "=" * 80)
    print("SUPERVISED MODELS - UNDERSAMPLING")
    print("=" * 80)
    X_train_under, y_train_under = balance_data(X_train, y_train, 'undersample')
    results_under = train_supervised_models(X_train_under, y_train_under, X_test, y_test, 'undersample')
    all_results.update({f"{k} (Undersample)": v for k, v in results_under.items()})
    
    # Step 5: Train unsupervised models
    print("\n" + "=" * 80)
    print("UNSUPERVISED MODELS")
    print("=" * 80)
    
    # Get normal data for training (reset indices to avoid issues)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_train_normal = X_train[y_train == 0]
    
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_val_normal = X_val[y_val == 0] if (y_val == 0).sum() > 0 else None
    
    # Test different contamination levels
    for contam in [0.01, 0.02]:
        print(f"\n--- Contamination: {contam} ---")
        results_unsup = train_unsupervised_models(X_train_normal, X_test, y_test, contam)
        all_results.update({f"{k} (cont={contam})": v for k, v in results_unsup.items()})
    
    # Step 6: Train autoencoder
    print("\n" + "=" * 80)
    print("DEEP LEARNING - AUTOENCODER")
    print("=" * 80)
    
    if TENSORFLOW_AVAILABLE:
        try:
            ae_result = train_autoencoder(X_train_normal, X_val_normal, X_test, X_train.shape[1])
            all_results['Autoencoder'] = ae_result
        except Exception as e:
            print(f"Error training autoencoder: {e}")
    else:
        print("Skipping autoencoder training: TensorFlow/Keras not available.")
        print("To enable autoencoder, install TensorFlow: pip install tensorflow")
    
    # Step 7: Calculate metrics and create plots
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    metrics_list = []
    
    for model_name, result in all_results.items():
        if 'probabilities' in result:
            metrics = calculate_metrics(y_test, result['predictions'], result['probabilities'])
        elif 'scores' in result:
            metrics = calculate_metrics(y_test, result['predictions'], result['scores'])
        else:
            metrics = calculate_metrics(y_test, result['predictions'])
        
        metrics['model_name'] = model_name
        metrics_list.append(metrics)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            model_name,
            f"reports/figures/cm_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        )
        
        # Save model
        if 'model' in result:
            model_file = f"models/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
            if 'keras' in str(type(result['model'])):
                result['model'].save(f"models/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.h5")
            else:
                joblib.dump(result['model'], model_file)
    
    # Create results dataframe
    results_df = pd.DataFrame(metrics_list)
    results_df.to_csv(f"{args.output}/results_summary.csv", index=False)
    print("\nResults saved to results_summary.csv")
    
    # Plot ROC curves for supervised models
    supervised_results = {k: v for k, v in all_results.items() if 'probabilities' in v}
    if supervised_results:
        plot_roc_curves(supervised_results, y_test, "reports/figures/roc_curves.png")
        plot_pr_curves(supervised_results, y_test, "reports/figures/pr_curves.png")
    
    # Plot imbalance comparison
    plot_imbalance_comparison(metrics_list, "reports/figures/imbalance_comparison.png")
    
    # Plot precision/recall vs threshold for unsupervised models
    for model_name, result in all_results.items():
        if 'scores' in result and 'probabilities' not in result:
            # This is an unsupervised model with scores
            plot_precision_recall_vs_threshold(
                y_test, result['scores'], model_name,
                f"reports/figures/pr_vs_threshold_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}.png"
            )
    
    # Find best model
    best_model = results_df.loc[results_df['f1_score'].idxmax()]
    print(f"\nBest Model: {best_model['model_name']}")
    print(f"F1 Score: {best_model['f1_score']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
