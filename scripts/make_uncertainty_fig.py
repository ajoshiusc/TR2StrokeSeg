import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter, zoom
import os

try:
    from nilearn.datasets import load_mni152_template
    has_nilearn = True
except ImportError:
    has_nilearn = False

def create_synthetic_lesion(shape, center, radius, noise_level=0.1):
    z, y, x = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    # Give it an irregular shape
    noise = np.random.normal(0, noise_level, shape)
    base = np.exp(- (dist**2) / (2 * radius**2)) + noise
    base = gaussian_filter(base, sigma=2)
    # Normalize to 0-1 (representing probability)
    base = np.clip(base, 0, None)
    if base.max() > 0:
        base = base / base.max()
    return base

def main():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'figure.dpi': 300,
        'axes.titleweight': 'bold',
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), constrained_layout=True)
    
    if has_nilearn:
        mni = load_mni152_template()
        mni_data = mni.get_fdata()
        # Take an axial slice
        slice_idx = mni_data.shape[2] // 2 + 5
        bg_slice = np.rot90(mni_data[:, :, slice_idx])
        shape = mni_data.shape
    else:
        # Fallback dummy brain
        shape = (100, 100, 100)
        bg_slice = np.zeros((100, 100))
        slice_idx = 50

    # Define centers (x, y, z) in terms of the array
    # Note: coordinates will be indexed as [x, y, slice]
    lesions = [
        {"name": "Small Lesion", "center": (shape[0]//2 - 15, shape[1]//2 + 10, slice_idx), "radius": 4},
        {"name": "Medium Lesion", "center": (shape[0]//2 + 20, shape[1]//2 - 10, slice_idx), "radius": 8},
        {"name": "Large Lesion", "center": (shape[0]//2 - 10, shape[1]//2 - 20, slice_idx), "radius": 15}
    ]

    np.random.seed(42)

    # Uncertainty colormap (coolwarm or similar)
    cmap = sns.color_palette("rocket", as_cmap=True)

    for i, (ax, l_info) in enumerate(zip(axes, lesions)):
        ax.imshow(bg_slice, cmap='gray', origin='lower')
        
        # Create probabilistic lesion
        prob_vol = create_synthetic_lesion(shape, l_info["center"], l_info["radius"])
        prob_slice = np.rot90(prob_vol[:, :, slice_idx])
        
        # We want to show quantile zones (e.g., 0.2, 0.4, 0.6, 0.8)
        levels = [0.2, 0.4, 0.6, 0.8]
        colors = ["#fcae91", "#fb6a4a", "#de2d26", "#a50f15"] # Reds showing confidence
        
        # Overlay quantiles as filled contours
        xx, yy = np.meshgrid(np.arange(prob_slice.shape[1]), np.arange(prob_slice.shape[0]))
        contour = ax.contourf(xx, yy, prob_slice, levels=levels + [1.0], colors=colors, alpha=0.7)
        ax.contour(xx, yy, prob_slice, levels=levels, colors="white", linewidths=0.5, alpha=0.8)
        
        ax.set_title(l_info["name"], pad=10)
        ax.axis('off')
        
        # Zoom in on the lesion for better visibility
        # Find center of mass of the prob_slice
        cy, cx = np.unravel_index(np.argmax(prob_slice), prob_slice.shape)
        window = max(25, l_info["radius"] * 4)
        ax.set_xlim(cx - window, cx + window)
        ax.set_ylim(cy - window, cy + window)

    # Add a colorbar indicating confidence
    # cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    # ... Wait, constrained_layout might conflict with manual axes. Use standard colorbar.
    fig.colorbar(contour, ax=axes, orientation='horizontal', fraction=0.08, pad=0.04, 
                 label='Model Confidence Zones (Quantiles)',
                 ticks=[0.3, 0.5, 0.7, 0.9])
    
    # Adjust ticks on colorbar to represent zones
    # But for a simpler layout, we just save.
    
    out_path = "segmentation_uncertainty_r01.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
