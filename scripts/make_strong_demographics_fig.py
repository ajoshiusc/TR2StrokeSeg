import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

def generate_strong_demographics_figure(output_path="strong_demographics_r01.png"):
    """
    Generates a compact, highly readable 1x5 multi-panel demographic figure
    suitable for an NIH R01 Research Strategy section focused on motor deficit prediction.
    """
    
    # Configure Matplotlib for NIH R01 standards
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,           
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'normal',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,         
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
    })

    # ==========================================================
    # 1. LOAD OR SIMULATE DATA
    # ==========================================================
    np.random.seed(42)
    n_subjects = 450
    df = pd.DataFrame({
        'Age': np.random.normal(loc=65, scale=12, size=n_subjects).clip(25, 95),
        'NIHSS': np.random.gamma(shape=2.0, scale=4.0, size=n_subjects).clip(0, 42).astype(int),
        'Lesion Volume (mL)': np.random.lognormal(mean=3.0, sigma=1.2, size=n_subjects).clip(0.1, 200),
        'Grip Ratio': np.random.beta(a=3, b=5, size=n_subjects), 
        'FMA-UE': np.random.normal(loc=35, scale=18, size=n_subjects).clip(0, 66).astype(int)
    })
    
    # ==========================================================
    # 2. CREATE THE FIGURE
    # Slightly taller figure (2.5 inches) for label clearance
    # ==========================================================
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.5))
    sns.set_theme(style="ticks")
    
    c_age = "#4C72B0"
    c_nihss = "#55A868"
    c_vol = "#C44E52"
    c_grip = "#8172B3"
    c_fma = "#CCB974"

    # Define titles and plotting settings
    plots_info = [
        {'x': 'Age', 'label': 'Years', 'title': 'Age', 'color': c_age, 'bins': 15, 'log': False},
        {'x': 'NIHSS', 'label': 'Score', 'title': 'NIHSS', 'color': c_nihss, 'bins': 15, 'log': False},
        {'x': 'Lesion Volume (mL)', 'label': 'mL (log)', 'title': 'Volume', 'color': c_vol, 'bins': 20, 'log': True},
        {'x': 'Grip Ratio', 'label': 'Ratio (Aff/Unaff)', 'title': 'Grip Ratio', 'color': c_grip, 'bins': 15, 'log': False},
        {'x': 'FMA-UE', 'label': 'Score (0-66)', 'title': 'FMA-UE', 'color': c_fma, 'bins': 15, 'log': False}
    ]

    for i, (ax, info) in enumerate(zip(axes, plots_info)):
        sns.histplot(data=df, x=info['x'], bins=info['bins'], log_scale=info['log'], ax=ax, color=info['color'])
        
        # Proper subplot labeling without overlapping
        ax.set_title(chr(65+i), loc='left', fontweight='bold', pad=10)
        ax.set_title(info['title'], loc='center', pad=10)
        
        ax.set_xlabel(info['label'])
        
        if i == 0:
            ax.set_ylabel('Count')
        else:
            ax.set_ylabel('')
            
        # Styling enhancements
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        sns.despine(ax=ax)
        
        # Restrict the maximum number of x-ticks to prevent horizontal overlap
        if info['log']:
            pass # Keep log locator
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=False))
            
    fig.tight_layout(w_pad=0.8)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved successfully to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_strong_demographics_figure()
