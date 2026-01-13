"""
Functions for evaluating models and creating plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)


def calculate_all_metrics(y_true, y_pred, y_proba=None):
    """Calculate all evaluation metrics"""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # False positive and false negative rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'fnr': fnr,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'confusion_matrix': cm
    }
    
    # ROC-AUC and PR-AUC if probabilities available
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    
    return metrics


def plot_confusion_matrix(cm, model_name, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_proba_list, model_names, save_path):
    """Plot ROC curves for multiple models"""
    plt.figure(figsize=(10, 8))
    
    for i, y_proba in enumerate(y_proba_list):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f'{model_names[i]} (AUC={auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to {save_path}")


def plot_pr_curve(y_true, y_proba_list, model_names, save_path):
    """Plot Precision-Recall curves"""
    plt.figure(figsize=(10, 8))
    
    for i, y_proba in enumerate(y_proba_list):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, label=f'{model_names[i]} (AP={pr_auc:.3f})', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PR curves to {save_path}")


def plot_anomaly_scores(normal_scores, fraud_scores, model_name, threshold=None, save_path=None):
    """Plot distribution of anomaly scores"""
    plt.figure(figsize=(12, 6))
    
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    plt.hist(fraud_scores, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    
    if threshold is not None:
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Anomaly Score Distribution - {model_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved anomaly scores plot to {save_path}")
    else:
        plt.close()


def create_results_table(metrics_list, save_path):
    """Create a summary table of all results"""
    df = pd.DataFrame(metrics_list)
    
    # Select columns to display
    display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score',
                   'roc_auc', 'pr_auc', 'fpr', 'fnr', 'fp', 'fn']
    display_cols = [col for col in display_cols if col in df.columns]
    
    df_display = df[display_cols].copy()
    
    # Round numeric columns
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(4)
    
    print("\n=== Results Summary ===")
    print(df_display.to_string(index=False))
    
    if save_path:
        df_display.to_csv(save_path, index=False)
        print(f"\nSaved results to {save_path}")
    
    return df_display
