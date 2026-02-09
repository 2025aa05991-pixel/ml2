"""
Complete training pipeline for Breast Cancer Wisconsin (Diagnostic) Dataset
Trains 6 models and computes 6 metrics for each
Author: BITS ML Assignment 2
Date: February 2026
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import joblib
import json
import os
from pathlib import Path

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODELS_DIR = Path(__file__).parent / "saved_models"
METRICS_FILE = Path(__file__).parent / "metrics_comparison.csv"
SCHEMA_FILE = Path(__file__).parent / "expected_schema.json"
SAMPLE_FILE = Path(__file__).parent.parent / "sample_test.csv"


def load_data():
    """Load Breast Cancer Wisconsin dataset"""
    print("=" * 80)
    print("LOADING BREAST CANCER WISCONSIN (DIAGNOSTIC) DATASET")
    print("=" * 80)
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='diagnosis')
    
    # Convert to original labels: 0=malignant (M), 1=benign (B)
    # Original dataset: M=malignant (positive class), B=benign (negative class)
    y_labels = y.map({0: 'M', 1: 'B'})
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Instances: {len(X)}")
    print(f"  - Features: {len(X.columns)}")
    print(f"  - Target distribution:")
    print(f"    Malignant (M): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"    Benign (B): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    print(f"  - Missing values: {X.isnull().sum().sum()}")
    
    # Save schema for app validation
    schema = {
        "n_features": len(X.columns),
        "feature_names": X.columns.tolist(),
        "feature_types": {col: str(X[col].dtype) for col in X.columns},
        "target_name": "diagnosis",
        "target_classes": ["M", "B"]
    }
    
    os.makedirs(MODELS_DIR.parent, exist_ok=True)
    with open(SCHEMA_FILE, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"\n✓ Schema saved to {SCHEMA_FILE}")
    
    # Create sample test CSV (10 random samples)
    sample_X = X.sample(n=10, random_state=RANDOM_STATE)
    sample_X.to_csv(SAMPLE_FILE, index=False)
    print(f"✓ Sample test CSV saved to {SAMPLE_FILE}")
    
    return X, y, y_labels


def check_class_imbalance(y):
    """Check if class imbalance requires special handling"""
    pos_count = (y == 0).sum()  # Malignant
    neg_count = (y == 1).sum()  # Benign
    total = len(y)
    imbalance_ratio = abs(pos_count - neg_count) / total
    
    needs_balancing = imbalance_ratio > 0.20
    
    print(f"\n{'=' * 80}")
    print("CLASS IMBALANCE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"  Imbalance ratio: {imbalance_ratio:.4f}")
    print(f"  Threshold: 0.20")
    print(f"  Needs balancing: {'YES' if needs_balancing else 'NO'}")
    
    return needs_balancing


def create_models(use_class_weight=False):
    """Create all 6 models with proper configuration"""
    models = {}
    
    # 1. Logistic Regression (with StandardScaler)
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=10000,
            class_weight='balanced' if use_class_weight else None
        ))
    ])
    models['Logistic Regression'] = lr_pipeline
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced' if use_class_weight else None
    )
    models['Decision Tree'] = dt
    
    # 3. KNN (with StandardScaler)
    knn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])
    models['KNN'] = knn_pipeline
    
    # 4. Naive Bayes (Gaussian for numeric features)
    nb = GaussianNB()
    models['Naive Bayes'] = nb
    
    # 5. Random Forest
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        class_weight='balanced' if use_class_weight else None
    )
    models['Random Forest'] = rf
    
    # 6. XGBoost
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=100
    )
    # XGBoost handles class imbalance with scale_pos_weight
    if use_class_weight:
        # Calculate scale_pos_weight for binary classification
        # This will be set during training after we know the class distribution
        pass
    models['XGBoost'] = xgb
    
    return models


def compute_metrics(y_true, y_pred, y_pred_proba):
    """Compute all 6 required metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba[:, 1]) if y_pred_proba.shape[1] == 2 else roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0),
        'F1': f1_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def train_and_evaluate():
    """Main training and evaluation pipeline"""
    # Load data
    X, y, y_labels = load_data()
    
    # Check class imbalance
    use_class_weight = check_class_imbalance(y)
    
    # Train-test split with stratification
    print(f"\n{'=' * 80}")
    print("TRAIN-TEST SPLIT")
    print(f"{'=' * 80}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train size: {len(X_train)} ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"  Test size: {len(X_test)} ({TEST_SIZE*100:.0f}%)")
    
    # Create models
    models = create_models(use_class_weight)
    
    # Train and evaluate each model
    print(f"\n{'=' * 80}")
    print("TRAINING AND EVALUATION")
    print(f"{'=' * 80}")
    
    results = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n[{model_name}]")
        print(f"  Training...")
        
        # Special handling for XGBoost class weight
        if model_name == 'XGBoost' and use_class_weight:
            neg_count = (y_train == 1).sum()
            pos_count = (y_train == 0).sum()
            scale_pos_weight = neg_count / pos_count
            model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  ✓ Trained successfully")
        print(f"  Metrics:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
        
        # Save model
        model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        print(f"  ✓ Saved to {model_path.name}")
        
        # Store results
        result = {'Model': model_name}
        result.update(metrics)
        results.append(result)
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    
    # Round to 4 decimal places
    numeric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    df_results[numeric_cols] = df_results[numeric_cols].round(4)
    
    # Sort by F1 score descending
    df_results = df_results.sort_values('F1', ascending=False).reset_index(drop=True)
    
    # Save metrics
    df_results.to_csv(METRICS_FILE, index=False)
    
    # Print final comparison
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON TABLE (sorted by F1 score)")
    print(f"{'=' * 80}\n")
    print(df_results.to_string(index=False))
    
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"  ✓ All models saved to: {MODELS_DIR}")
    print(f"  ✓ Metrics saved to: {METRICS_FILE}")
    print(f"  ✓ Schema saved to: {SCHEMA_FILE}")
    print(f"  ✓ Sample CSV saved to: {SAMPLE_FILE}")
    print(f"\nNext step: Run the Streamlit app with 'streamlit run app.py'")
    
    return df_results


if __name__ == "__main__":
    train_and_evaluate()
