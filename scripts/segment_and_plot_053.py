import os
import shutil
import subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from scipy.stats import entropy
import sys

def main():
    # Setup environment
    os.environ["nnUNet_raw"] = "/home/ajoshi/project2_ajoshi_1183/data/TR2/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = "/home/ajoshi/project2_ajoshi_1183/data/TR2/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "/home/ajoshi/project2_ajoshi_1183/data/TR2/nnUNet_results"

    # Define relative paths based on repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_path = os.path.join(repo_root, "nnUNet_raw/Dataset001_Atlas2/imagesTr/Atlas2_053_0000.nii.gz")
    gt_path = os.path.join(repo_root, "nnUNet_raw/Dataset001_Atlas2/labelsTr/Atlas2_053.nii.gz")

    if not os.path.exists(img_path):
        print(f"Error: Could not find image at {img_path}")
        sys.exit(1)

    in_dir = os.path.join(repo_root, "intermediate_outputs", "test_053_in")
    out_dir = os.path.join(repo_root, "intermediate_outputs", "test_053_out")

    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Empty dirs to be safe
    for f in os.listdir(in_dir): os.remove(os.path.join(in_dir, f))
    for f in os.listdir(out_dir): os.remove(os.path.join(out_dir, f))

    # Copy image
    shutil.copy(img_path, os.path.join(in_dir, "Atlas2_053_0000.nii.gz"))

    print("Running nnUNet inference on Atlas2_053...")
    # Run inference with save probabilities
    cmd = [
        "nnUNetv2_predict", 
        "-i", in_dir, 
        "-o", out_dir, 
        "-d", "Dataset001_Atlas2",
        "-c", "3d_fullres", 
        "-f", "0", 
        "-chk", "checkpoint_final.pth",
        "--save_probabilities", 
        "-device", "cpu", 
        "--disable_progress_bar"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        print("Error running nnUNet predict:", e)
        sys.exit(1)

    print("Inference completed. Generating plots...")

    # Load results
    pred_path = os.path.join(out_dir, "Atlas2_053.nii.gz")
    npz_path = os.path.join(out_dir, "Atlas2_053.npz")

    t1_img = nib.load(img_path)
    gt_img = nib.load(gt_path)
    pred_img = nib.load(pred_path)
    probs = np.load(npz_path)['probabilities']

    # Calculate uncertainty as entropy of probability distribution over classes
    # probs shape is (classes, x, y, z) - here 2 classes (bg, lesion)
    uncertainty_map = entropy(probs, axis=0) / np.log(probs.shape[0]) # Normalized to 0-1
    uncertainty_img = nib.Nifti1Image(uncertainty_map, t1_img.affine, header=t1_img.header)

    # Find the best slice to visualize (e.g., largest lesion area in ground truth)
    gt_data = gt_img.get_fdata()
    slice_idx = np.argmax(np.sum(gt_data, axis=(0,1)))
    coords = nib.affines.apply_affine(gt_img.affine, [gt_data.shape[0]//2, gt_data.shape[1]//2, slice_idx])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("ATLAS2 Case 053 - Lesion Segmentation & Uncertainty", fontsize=16)

    # Figure 1: Outlines (Manual + Auto)
    display1 = plotting.plot_anat(
        t1_img, 
        display_mode='z', 
        cut_coords=[coords[2]], 
        axes=axes[0],
        title="Red: Auto | Green: Manual",
        dim=-0.5
    )
    display1.add_contours(gt_img, levels=[0.5], colors=['green'], linewidths=2.5)
    display1.add_contours(pred_img, levels=[0.5], colors=['red'], linewidths=2.5)

    # Figure 2: Uncertainty Map
    display2 = plotting.plot_anat(
        t1_img, 
        display_mode='z', 
        cut_coords=[coords[2]], 
        axes=axes[1],
        title="Uncertainty Colormap",
        dim=-0.5
    )
    
    # Check max uncertainty to ensure our colormap scales appropriately
    max_u = np.max(uncertainty_map)
    print(f"Max uncertainty value in map: {max_u:.4f}")
    
    # Lowering the vmin threshold to make slightly uncertain regions visible
    # Removing 'transparency' or 'alpha' directly so colors are vibrant
    display2.add_overlay(uncertainty_img, cmap='hot', vmin=0.01, vmax=max_u)

    out_plot = os.path.join(repo_root, "Atlas2_053_segmentation_overlay.png")
    plt.savefig(out_plot, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Plot saved to: {out_plot}")
    plt.show()

if __name__ == "__main__":
    main()
