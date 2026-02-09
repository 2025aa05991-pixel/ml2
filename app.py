"""
Breast Cancer Prediction - Streamlit Application
BITS ML Assignment 2
Author: Vivek Chaudhary
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "model" / "saved_models"
METRICS_FILE = BASE_DIR / "model" / "metrics_comparison.csv"
SCHEMA_FILE = BASE_DIR / "model" / "expected_schema.json"
SAMPLE_FILE = BASE_DIR / "sample_test.csv"

# Model names mapping
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}


@st.cache_data
def load_schema():
    """Load expected schema"""
    if not SCHEMA_FILE.exists():
        st.error(f"Schema file not found at {SCHEMA_FILE}. Please run model/train_all.py first.")
        st.stop()
    
    with open(SCHEMA_FILE, 'r') as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    """Load pre-computed metrics comparison"""
    if not METRICS_FILE.exists():
        st.warning("Metrics file not found. Please run model/train_all.py first.")
        return None
    
    return pd.read_csv(METRICS_FILE)


@st.cache_resource
def load_model(model_name):
    """Load a trained model"""
    model_file = MODELS_DIR / MODEL_FILES[model_name]
    
    if not model_file.exists():
        st.error(f"Model file not found: {model_file}. Please run model/train_all.py first.")
        st.stop()
    
    return joblib.load(model_file)


def validate_input_data(df, schema):
    """Validate uploaded CSV against expected schema"""
    errors = []
    warnings = []
    
    # Check number of features
    if len(df.columns) != schema['n_features']:
        errors.append(f"Expected {schema['n_features']} features, got {len(df.columns)}")
    
    # Check feature names
    expected_features = set(schema['feature_names'])
    actual_features = set(df.columns)
    
    missing_features = expected_features - actual_features
    if missing_features:
        errors.append(f"Missing features: {', '.join(sorted(missing_features))}")
    
    extra_features = actual_features - expected_features
    if extra_features:
        warnings.append(f"Extra features (will be ignored): {', '.join(sorted(extra_features))}")
    
    # Check for missing values
    if df[schema['feature_names']].isnull().any().any():
        missing_counts = df[schema['feature_names']].isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        warnings.append(f"Missing values detected in: {', '.join(missing_features.index.tolist())}")
    
    return errors, warnings


def make_predictions(model, X):
    """Make predictions with error handling"""
    try:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Convert predictions to labels (0='M' malignant, 1='B' benign)
        labels = np.array(['M', 'B'])
        y_pred_labels = labels[y_pred]
        
        return y_pred, y_pred_proba, y_pred_labels
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()


def display_metrics(model_name, metrics_df):
    """Display model metrics in a clean format"""
    model_metrics = metrics_df[metrics_df['Model'] == model_name]
    
    if model_metrics.empty:
        st.warning(f"No metrics found for {model_name}")
        return
    
    # Extract metrics
    metrics = model_metrics.iloc[0]
    
    # Display in columns
    st.subheader(f"📊 {model_name} Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.metric("AUC", f"{metrics['AUC']:.4f}")
    
    with col2:
        st.metric("Precision", f"{metrics['Precision']:.4f}")
        st.metric("Recall", f"{metrics['Recall']:.4f}")
    
    with col3:
        st.metric("F1 Score", f"{metrics['F1']:.4f}")
        st.metric("MCC", f"{metrics['MCC']:.4f}")


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Malignant (M)', 'Benign (B)']
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    # Header
    st.title("🔬 Breast Cancer Wisconsin (Diagnostic) Prediction")
    st.markdown("**BITS ML Assignment 2** | Six Models Comparison")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application demonstrates binary classification on the 
        **Breast Cancer Wisconsin (Diagnostic)** dataset using 6 machine learning models:
        
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors (KNN)
        4. Naive Bayes (Gaussian)
        5. Random Forest
        6. XGBoost
        
        **Dataset:** 569 instances, 30 features  
        **Target:** Malignant (M) vs Benign (B)  
        **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
        """)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Download sample CSV")
        st.markdown("2. Upload your test data")
        st.markdown("3. Select a model")
        st.markdown("4. View predictions & metrics")
    
    # Load data
    schema = load_schema()
    metrics_df = load_metrics()
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["🔮 Predictions", "📈 Model Comparison", "📥 Download Sample"])
    
    # TAB 1: Predictions
    with tab1:
        st.header("Make Predictions")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model:",
            list(MODEL_FILES.keys()),
            help="Choose one of the 6 trained models"
        )
        
        # File upload
        st.markdown("### Upload Test Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file with the same 30 features as training data",
            type=['csv'],
            help="CSV must contain all 30 feature columns"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✓ File uploaded: {uploaded_file.name}")
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Validate
                errors, warnings = validate_input_data(df, schema)
                
                if errors:
                    st.error("**Validation Errors:**")
                    for error in errors:
                        st.error(f"❌ {error}")
                    st.stop()
                
                if warnings:
                    st.warning("**Warnings:**")
                    for warning in warnings:
                        st.warning(f"⚠️ {warning}")
                
                # Show preview
                with st.expander("👁️ Preview Data (first 5 rows)", expanded=False):
                    st.dataframe(df.head(), width='stretch')
                
                # Ensure correct column order
                X = df[schema['feature_names']]
                
                # Load model and predict
                model = load_model(selected_model)
                
                with st.spinner(f"Running {selected_model}..."):
                    y_pred, y_pred_proba, y_pred_labels = make_predictions(model, X)
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Index': range(len(y_pred)),
                    'Prediction': y_pred_labels,
                    'Confidence (Malignant)': y_pred_proba[:, 0],
                    'Confidence (Benign)': y_pred_proba[:, 1]
                })
                
                # Summary statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Samples", len(y_pred))
                    st.metric("Predicted Malignant (M)", (y_pred == 0).sum())
                
                with col2:
                    st.metric("Predicted Benign (B)", (y_pred == 1).sum())
                    st.metric("Avg Confidence", f"{y_pred_proba.max(axis=1).mean():.2%}")
                
                # Show results table
                st.markdown("### Detailed Predictions")
                st.dataframe(
                    results_df.style.background_gradient(
                        subset=['Confidence (Malignant)', 'Confidence (Benign)'],
                        cmap='RdYlGn_r'
                    ),
                    width='stretch'
                )
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Show metrics for selected model
                if metrics_df is not None:
                    st.markdown("---")
                    display_metrics(selected_model, metrics_df)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV matches the expected format.")
        
        else:
            st.info("👆 Upload a CSV file to start making predictions")
    
    # TAB 2: Model Comparison
    with tab2:
        st.header("Model Performance Comparison")
        
        if metrics_df is not None:
            st.markdown("""
            This table shows the performance of all 6 models on the test set.  
            **Sorted by F1 Score (descending)**
            """)
            
            # Display full comparison table
            st.dataframe(
                metrics_df.style.background_gradient(
                    subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                    cmap='YlGn'
                ).format({
                    'Accuracy': '{:.4f}',
                    'AUC': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1': '{:.4f}',
                    'MCC': '{:.4f}'
                }),
                width='stretch'
            )
            
            # Download button
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics Comparison",
                data=csv,
                file_name="metrics_comparison.csv",
                mime="text/csv"
            )
            
            # Visualization
            st.markdown("---")
            st.subheader("📊 Visual Comparison")
            
            # Select metrics to plot
            metrics_to_plot = st.multiselect(
                "Select metrics to visualize:",
                ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                default=['Accuracy', 'F1', 'AUC']
            )
            
            if metrics_to_plot:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(metrics_df))
                width = 0.8 / len(metrics_to_plot)
                
                for i, metric in enumerate(metrics_to_plot):
                    offset = (i - len(metrics_to_plot)/2) * width + width/2
                    ax.bar(x + offset, metrics_df[metric], width, label=metric, alpha=0.8)
                
                ax.set_xlabel('Models', fontweight='bold')
                ax.set_ylabel('Score', fontweight='bold')
                ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, 1.05)
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Metrics file not found. Please run `python model/train_all.py` first.")
    
    # TAB 3: Download Sample
    with tab3:
        st.header("Download Sample Test Data")
        
        st.markdown("""
        Download a sample CSV file that matches the expected schema.  
        Use this as a template for your own test data.
        
        **Expected Features:** 30 numeric features from the Breast Cancer Wisconsin dataset
        """)
        
        if SAMPLE_FILE.exists():
            with open(SAMPLE_FILE, 'rb') as f:
                st.download_button(
                    label="📥 Download sample_test.csv",
                    data=f,
                    file_name="sample_test.csv",
                    mime="text/csv",
                    help="10 sample instances with all 30 features"
                )
            
            # Show preview
            sample_df = pd.read_csv(SAMPLE_FILE)
            st.markdown("### Preview (first 5 rows)")
            st.dataframe(sample_df.head(), width='stretch')
            st.info(f"Sample contains {len(sample_df)} instances with {len(sample_df.columns)} features")
        
        else:
            st.warning("Sample file not found. Run `python model/train_all.py` to generate it.")

        # Schema information
        st.markdown("---")
        st.markdown("### 📋 Expected Schema")
        
        with st.expander("View feature names", expanded=False):
            st.json(schema['feature_names'])


if __name__ == "__main__":
    main()
