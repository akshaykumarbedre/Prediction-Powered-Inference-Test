#!/usr/bin/env python3
"""
Example script demonstrating Prediction-Powered Inference concepts
without requiring the full ppi_py library.

This provides a simplified demonstration of PPI methodology using 
basic statistical techniques to show the core concepts.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt


def classical_mean_ci(data, alpha=0.05):
    """
    Compute classical confidence interval for the mean.
    
    Parameters:
    -----------
    data : array-like
        Sample data
    alpha : float
        Significance level (default: 0.05 for 95% CI)
        
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    # Use t-distribution for small samples
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_critical * std_err
    
    return (mean - margin_error, mean + margin_error)


def simple_ppi_mean_ci(labeled_data, labeled_predictions, unlabeled_predictions, alpha=0.05):
    """
    Simplified Prediction-Powered Inference for mean estimation.
    
    This is a basic implementation showing PPI concepts, not the full method.
    
    Parameters:
    -----------
    labeled_data : array-like
        True labels for the labeled subset
    labeled_predictions : array-like  
        Model predictions for the labeled subset
    unlabeled_predictions : array-like
        Model predictions for the unlabeled subset
    alpha : float
        Significance level
        
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    # Step 1: Estimate bias using labeled data
    bias = np.mean(labeled_data) - np.mean(labeled_predictions)
    
    # Step 2: Bias-corrected estimate using all predictions
    all_predictions = np.concatenate([labeled_predictions, unlabeled_predictions])
    ppi_estimate = np.mean(all_predictions) + bias
    
    # Step 3: Estimate variance (simplified approach)
    # In full PPI, this would use more sophisticated variance estimation
    labeled_residuals = labeled_data - labeled_predictions
    bias_var = np.var(labeled_residuals) / len(labeled_data)
    prediction_var = np.var(all_predictions) / len(all_predictions)
    
    # Combine variances (simplified)
    total_var = bias_var + prediction_var * (len(unlabeled_predictions) / len(all_predictions))**2
    std_err = np.sqrt(total_var)
    
    # Construct confidence interval
    t_critical = stats.t.ppf(1 - alpha/2, df=len(labeled_data)-1)
    margin_error = t_critical * std_err
    
    return (ppi_estimate - margin_error, ppi_estimate + margin_error)


def run_comparison_demo(n_total=2000, n_labeled_range=(50, 500), n_trials=50):
    """
    Run a comparison demo between Classical and simplified PPI methods.
    """
    print("=== Simplified PPI vs Classical Inference Demo ===\n")
    
    # Generate synthetic dataset
    X, y = make_regression(n_samples=n_total, n_features=8, noise=0.1, random_state=42)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    true_mean = np.mean(y_test)
    
    print(f"Dataset: {n_total} total samples")
    print(f"True population mean: {true_mean:.4f}")
    print(f"Model R² score: {model.score(X_test, y_test):.4f}\n")
    
    # Test different labeled sample sizes
    results = []
    
    for n_labeled in range(n_labeled_range[0], n_labeled_range[1]+1, 50):
        for trial in range(n_trials):
            # Random sample for this trial
            indices = np.random.permutation(len(y_test))
            labeled_idx = indices[:n_labeled]
            unlabeled_idx = indices[n_labeled:]
            
            # Split data
            y_labeled = y_test[labeled_idx]
            y_pred_labeled = y_pred[labeled_idx]
            y_pred_unlabeled = y_pred[unlabeled_idx]
            
            # Classical CI (labeled data only)
            classical_ci = classical_mean_ci(y_labeled)
            classical_width = classical_ci[1] - classical_ci[0]
            classical_covers = classical_ci[0] <= true_mean <= classical_ci[1]
            
            # Simplified PPI CI
            ppi_ci = simple_ppi_mean_ci(y_labeled, y_pred_labeled, y_pred_unlabeled)
            ppi_width = ppi_ci[1] - ppi_ci[0]
            ppi_covers = ppi_ci[0] <= true_mean <= ppi_ci[1]
            
            # Store results
            results.append({
                'n_labeled': n_labeled,
                'trial': trial,
                'method': 'Classical',
                'lower': classical_ci[0],
                'upper': classical_ci[1], 
                'width': classical_width,
                'covers': classical_covers
            })
            
            results.append({
                'n_labeled': n_labeled,
                'trial': trial,
                'method': 'PPI',
                'lower': ppi_ci[0],
                'upper': ppi_ci[1],
                'width': ppi_width,
                'covers': ppi_covers
            })
    
    df = pd.DataFrame(results)
    
    # Analyze results
    print("Results Summary:")
    print("-" * 40)
    
    summary = df.groupby(['method', 'n_labeled']).agg({
        'width': ['mean', 'std'],
        'covers': 'mean'
    }).round(4)
    
    print(summary)
    print()
    
    # Overall comparison
    overall = df.groupby('method').agg({
        'width': 'mean',
        'covers': 'mean'
    }).round(4)
    
    print("Overall Comparison:")
    print(overall)
    
    # Calculate improvement
    ppi_width = overall.loc['PPI', 'width']
    classical_width = overall.loc['Classical', 'width']
    improvement = (classical_width - ppi_width) / classical_width * 100
    
    print(f"\nPPI average width reduction: {improvement:.1f}%")
    
    # Create visualization
    create_comparison_plot(df, true_mean)
    
    return df


def create_comparison_plot(df, true_mean):
    """Create visualization comparing the methods."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Width vs sample size
    summary = df.groupby(['method', 'n_labeled'])['width'].mean().reset_index()
    
    for method in ['Classical', 'PPI']:
        method_data = summary[summary['method'] == method]
        axes[0].plot(method_data['n_labeled'], method_data['width'], 
                    'o-', label=method, linewidth=2, markersize=4)
    
    axes[0].set_xlabel('Labeled Sample Size')
    axes[0].set_ylabel('Average CI Width')
    axes[0].set_title('Confidence Interval Width vs Sample Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Coverage probability
    coverage = df.groupby(['method', 'n_labeled'])['covers'].mean().reset_index()
    
    for method in ['Classical', 'PPI']:
        method_data = coverage[coverage['method'] == method]
        axes[1].plot(method_data['n_labeled'], method_data['covers'], 
                    'o-', label=method, linewidth=2, markersize=4)
    
    axes[1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Nominal 95%')
    axes[1].set_xlabel('Labeled Sample Size')
    axes[1].set_ylabel('Coverage Probability')
    axes[1].set_title('Coverage Probability vs Sample Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.8, 1.0)
    
    plt.tight_layout()
    plt.savefig('ppi_demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'ppi_demo_results.png'")


if __name__ == "__main__":
    # Run the demo
    np.random.seed(42)  # For reproducibility
    
    print("Running simplified PPI demonstration...")
    print("This shows the core concepts without requiring the full ppi_py library.\n")
    
    try:
        results_df = run_comparison_demo()
        print("\n✓ Demo completed successfully!")
        print("\nThis demonstrates how PPI can reduce confidence interval widths")
        print("while maintaining proper coverage by leveraging predictions on unlabeled data.")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure you have numpy, pandas, scikit-learn, scipy, and matplotlib installed.")