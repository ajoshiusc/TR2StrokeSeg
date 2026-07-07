import os
import glob
import subprocess
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
from nilearn.datasets import load_mni152_template

def get_subjects():
    strong_dir = "/deneb_disk/STRONG_neuroimages_to_share/"
    subjects = sorted(glob.glob(os.path.join(strong_dir, "sub-*")))
    sizes = []
    
    for sub in subjects:
        mask_path = glob.glob(os.path.join(sub, "*_mask_mni_1mm.nii.gz"))
        mni_path = [p for p in glob.glob(os.path.join(sub, "*_mni_1mm.nii.gz")) if "mask" not in p]
        if mask_path and mni_path:
            mask = nib.load(mask_path[0]).get_fdata()
            vol = np.sum(mask > 0)
            if vol > 0:
                sizes.append((os.path.basename(sub), vol, mni_path[0]))
    
    sizes.sort(key=lambda x: x[1])
    n = len(sizes)
    p10 = n // 10
    small = sizes[p10 : p10+5]
    med_idx = n // 2
    medium = sizes[med_idx-2 : med_idx+3]
    p90 = int(n * 0.9)
    large = sizes[p90-5 : p90]
    return small, medium, large

def run_nnunet(subjects, category_name):
    in_dir = f"/tmp/nnunet_in_{category_name}"
    out_dir = f"/tmp/nnunet_out_{category_name}"
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(in_dir + "/*"): os.remove(f)
    for f in glob.glob(out_dir + "/*"): os.remove(f)
    for sub, idx, target_path in subjects:
        dest = os.path.join(in_dir, f"{sub}_0000.nii.gz")
        shutil.copy(target_path, dest)
    print(f"Running nnUNet for {category_name} subjects ({len(subjects)} scans)...")
    env = os.environ.copy()
    env["nnUNet_raw"] = "/tmp/nnUNet_raw"
    env["nnUNet_preprocessed"] = "/tmp/nnUNet_preprocessed"
    env["nnUNet_results"] = "/home/ajoshi/Desktop/backup_desktop/models/nnUNet_results"
    cmd = [
        "nnUNetv2_predict", "-i", in_dir, "-o", out_dir, "-d", "1", 
        "-c", "3d_fullres", "-f", "0", "-chk", "checkpoint_best.pth", 
        "--save_probabilities", "-device", "cpu", "--disable_tta"
    ]
    subprocess.run(cmd, env=env)
    return in_dir, out_dir

def plot_category(subjects, in_dir, out_dir, category_name):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
    fig.suptitle(f"{category_name.capitalize()} Lesions Candidates", fontweight='bold')
    levels = [0.2, 0.4, 0.6, 0.8]
    colors = ["#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    mni_template = load_mni152_template(resolution=1)
    
    for i, (sub, vol, npz_path) in enumerate(subjects):
        ax = axes[i]
        npz_file = os.path.join(out_dir, f"{sub}.npz")
        nii_pred = os.path.join(in_dir, f"{sub}_0000.nii.gz")
        if not os.path.exists(npz_file):
            ax.set_title(f"{sub}\nFailed Predict", fontsize=8)
            ax.axis('off')
            continue
            
        subj_img = nib.load(nii_pred)
        probs = np.load(npz_file)['probabilities']
        prob_mask_orig = probs[1]
        
        # The nnUNet returns (Z, Y, X), so we transpose to (X, Y, Z)
        prob_mask = prob_mask_orig.transpose(2, 1, 0)
        
        prob_img = nib.Nifti1Image(prob_mask, subj_img.affine, header=subj_img.header)
        max_idx = np.unravel_index(np.argmax(prob_mask), prob_mask.shape)
        coords = nib.affines.apply_affine(subj_img.affine, max_idx)
        
        display = plotting.plot_img(
            mni_template,
            display_mode="z",
            cut_coords=[coords[2]], 
            axes=ax,
            cmap="gray",
            annotate=False,
            draw_cross=False,
            title=f"{sub}\nVol: {vol}"
        )
        if np.max(prob_mask) > 0.2:
            display.add_contours(prob_img, levels=levels, colors=colors, linewidths=1.5, alpha=0.8)
            
    out_img = f"{category_name}_lesions_candidates.png"
    plt.savefig(out_img, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved {out_img}")

def main():
    print("Finding subjects...")
    small, med, large = get_subjects()
    print("Starting small...")
    in_dir, out_dir = run_nnunet(small, "small")
    plot_category(small, in_dir, out_dir, "small")
    print("Starting medium...")
    in_dir, out_dir = run_nnunet(med, "medium")
    plot_category(med, in_dir, out_dir, "medium")
    print("Starting large...")
    in_dir, out_dir = run_nnunet(large, "large")
    plot_category(large, in_dir, out_dir, "large")
    print("Done!")

if __name__ == "__main__":
    main()
