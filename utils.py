"""
Utility functions for Prediction-Powered Inference visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_plots(df, save_path, n_idx=3, intervals_xlabel="Median House Value", true_theta=None):
    """
    Create visualization plots comparing PPI, Classical, and Imputation methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing results with columns: method, n, lower, upper, width
    save_path : str
        Path to save the generated plot
    n_idx : int
        Index for selecting which sample size to display in the intervals plot (default: 3)
    intervals_xlabel : str
        Label for the x-axis of the intervals plot
    true_theta : float
        True parameter value to display as a reference line
    """
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Color scheme for different methods
    colors = {
        'PPI': '#2E86AB',
        'Classical': '#A23B72', 
        'Imputation': '#F18F01'
    }
    
    # Plot 1: Confidence interval widths vs sample size
    ax1 = axes[0]
    
    # Group by method and sample size, calculate mean width
    width_summary = df[df['method'] != 'Imputation'].groupby(['method', 'n'])['width'].agg(['mean', 'std']).reset_index()
    
    for method in ['PPI', 'Classical']:
        method_data = width_summary[width_summary['method'] == method]
        # Handle the case where width values might be arrays (from PPI results)
        means = []
        stds = []
        ns = []
        
        for _, row in method_data.iterrows():
            if isinstance(row['mean'], (list, np.ndarray)):
                means.append(row['mean'][0] if len(row['mean']) > 0 else 0)
                stds.append(row['std'][0] if len(row['std']) > 0 else 0)
            else:
                means.append(row['mean'])
                stds.append(row['std'])
            ns.append(row['n'])
        
        ax1.errorbar(ns, means, yerr=stds, label=method, 
                    color=colors[method], marker='o', linewidth=2, markersize=4)
    
    ax1.set_xlabel('Sample Size (n)')
    ax1.set_ylabel('Confidence Interval Width')
    ax1.set_title('Interval Width vs Sample Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coverage probability
    ax2 = axes[1]
    
    # Calculate coverage for each method and sample size
    coverage_data = []
    
    for method in ['PPI', 'Classical']:
        method_df = df[df['method'] == method]
        for n in method_df['n'].unique():
            if pd.isna(n):
                continue
            subset = method_df[method_df['n'] == n]
            
            # Check coverage for each trial
            coverage_trials = []
            for _, row in subset.iterrows():
                lower = row['lower']
                upper = row['upper']
                
                # Handle array format from PPI results
                if isinstance(lower, (list, np.ndarray)):
                    lower = lower[0] if len(lower) > 0 else 0
                if isinstance(upper, (list, np.ndarray)):
                    upper = upper[0] if len(upper) > 0 else 0
                
                # Check if true_theta is within the interval
                if true_theta is not None:
                    covers = lower <= true_theta <= upper
                    coverage_trials.append(covers)
            
            if coverage_trials:
                coverage = np.mean(coverage_trials)
                coverage_data.append({'method': method, 'n': n, 'coverage': coverage})
    
    coverage_df = pd.DataFrame(coverage_data)
    
    for method in ['PPI', 'Classical']:
        method_data = coverage_df[coverage_df['method'] == method]
        if not method_data.empty:
            ax2.plot(method_data['n'], method_data['coverage'], 
                    label=method, color=colors[method], marker='o', linewidth=2, markersize=4)
    
    # Add nominal coverage line
    ax2.axhline(y=0.95, color='black', linestyle='--', alpha=0.7, label='Nominal (95%)')
    
    ax2.set_xlabel('Sample Size (n)')
    ax2.set_ylabel('Coverage Probability')
    ax2.set_title('Coverage Probability vs Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.0)
    
    # Plot 3: Confidence intervals for a specific sample size
    ax3 = axes[2]
    
    # Select data for a specific sample size
    sample_sizes = sorted([n for n in df['n'].unique() if not pd.isna(n)])
    if n_idx < len(sample_sizes):
        selected_n = sample_sizes[n_idx]
        
        interval_data = df[df['n'] == selected_n]
        
        y_pos = 0
        method_positions = {}
        
        for method in ['PPI', 'Classical']:
            method_subset = interval_data[interval_data['method'] == method].head(10)  # Show first 10 intervals
            
            for _, row in method_subset.iterrows():
                lower = row['lower']
                upper = row['upper']
                
                # Handle array format
                if isinstance(lower, (list, np.ndarray)):
                    lower = lower[0] if len(lower) > 0 else 0
                if isinstance(upper, (list, np.ndarray)):
                    upper = upper[0] if len(upper) > 0 else 0
                
                # Draw confidence interval
                ax3.plot([lower, upper], [y_pos, y_pos], 
                        color=colors[method], linewidth=2, alpha=0.7)
                ax3.plot([lower, lower], [y_pos-0.1, y_pos+0.1], 
                        color=colors[method], linewidth=2)
                ax3.plot([upper, upper], [y_pos-0.1, y_pos+0.1], 
                        color=colors[method], linewidth=2)
                
                y_pos += 1
            
            method_positions[method] = y_pos - len(method_subset)/2 - 0.5
            y_pos += 1  # Space between methods
    
    # Add true value line if provided
    if true_theta is not None:
        ax3.axvline(x=true_theta, color='red', linestyle='--', linewidth=2, 
                   label=f'True Value ({true_theta:.3f})')
    
    ax3.set_xlabel(intervals_xlabel)
    ax3.set_ylabel('Trial Number')
    ax3.set_title(f'Confidence Intervals (n={selected_n})')
    
    # Add method labels
    for method, pos in method_positions.items():
        ax3.text(ax3.get_xlim()[0], pos, method, verticalalignment='center', 
                color=colors[method], fontweight='bold')
    
    if true_theta is not None:
        ax3.legend()
    
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {save_path}")


def summary_statistics(df):
    """
    Print summary statistics for the inference comparison.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing results with columns: method, n, lower, upper, width
    """
    
    print("=== Prediction-Powered Inference Results Summary ===\n")
    
    # Overall width comparison
    print("Average Confidence Interval Widths by Method:")
    print("-" * 50)
    
    for method in ['PPI', 'Classical', 'Imputation']:
        method_data = df[df['method'] == method]
        if not method_data.empty:
            # Handle array values in width column
            widths = []
            for w in method_data['width']:
                if isinstance(w, (list, np.ndarray)):
                    if len(w) > 0:
                        widths.append(w[0])
                else:
                    widths.append(w)
            
            if widths:
                avg_width = np.mean(widths)
                print(f"{method:12s}: {avg_width:.4f}")
    
    print()
    
    # Width reduction calculation
    ppi_widths = []
    classical_widths = []
    
    # Get matched pairs of PPI and Classical results
    sample_sizes = [n for n in df['n'].unique() if not pd.isna(n)]
    
    for n in sample_sizes:
        ppi_subset = df[(df['method'] == 'PPI') & (df['n'] == n)]
        classical_subset = df[(df['method'] == 'Classical') & (df['n'] == n)]
        
        for _, ppi_row in ppi_subset.iterrows():
            ppi_width = ppi_row['width']
            if isinstance(ppi_width, (list, np.ndarray)):
                ppi_width = ppi_width[0] if len(ppi_width) > 0 else 0
            
            # Find corresponding classical result
            classical_match = classical_subset[classical_subset['trial'] == ppi_row['trial']]
            if not classical_match.empty:
                classical_width = classical_match.iloc[0]['width']
                
                ppi_widths.append(ppi_width)
                classical_widths.append(classical_width)
    
    if ppi_widths and classical_widths:
        reduction = (np.mean(classical_widths) - np.mean(ppi_widths)) / np.mean(classical_widths) * 100
        print(f"Average Width Reduction (PPI vs Classical): {reduction:.1f}%")
        print(f"PPI intervals are on average {reduction:.1f}% narrower than classical intervals\n")
    
    # Sample size analysis
    print("Interval Width by Sample Size:")
    print("-" * 40)
    
    for n in sorted(sample_sizes):
        print(f"\nSample size n = {int(n)}:")
        for method in ['PPI', 'Classical']:
            method_data = df[(df['method'] == method) & (df['n'] == n)]
            if not method_data.empty:
                widths = []
                for w in method_data['width']:
                    if isinstance(w, (list, np.ndarray)):
                        if len(w) > 0:
                            widths.append(w[0])
                    else:
                        widths.append(w)
                
                if widths:
                    avg_width = np.mean(widths)
                    std_width = np.std(widths)
                    print(f"  {method:10s}: {avg_width:.4f} ± {std_width:.4f}")