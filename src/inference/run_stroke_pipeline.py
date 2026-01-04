import os
import glob
import subprocess
import nibabel as nib
import numpy as np
from pathlib import Path
from shutil import copyfile

def run_nnunet_predict(input_nii, output_dir, dataset_id=1, config="3d_fullres", fold=0, checkpoint=None):
    """
    Run nnUNet prediction for a single preprocessed file.
    """
    # nnUNet expects _0000 for single channel
    base = os.path.splitext(os.path.basename(input_nii))[0]
    if base.endswith('.nii'):
        base = os.path.splitext(base)[0]
    # Create a temp input dir for nnUNet
    import tempfile
    with tempfile.TemporaryDirectory() as temp_in:
        nnunet_in = os.path.join(temp_in, f"{base}_0000.nii.gz")
        copyfile(input_nii, nnunet_in)
        nnunet_out = os.path.join(output_dir, "nnunet_pred")
        os.makedirs(nnunet_out, exist_ok=True)
        nnunet_venv = os.environ.get("NNUNET_VENV", os.path.expanduser("~/.venv/bin"))
        #nnunet_cmd = os.path.join(nnunet_venv, "nnUNetv2_predict")
        nnunet_cmd = os.path.join("nnUNetv2_predict")

        cmd = [
                nnunet_cmd,
                "-i", temp_in,
                "-o", nnunet_out,
                "-d", str(dataset_id),
                "-c", config,
                "-f", str(fold),
                "-chk", checkpoint or "checkpoint_best.pth",
                "-device", "cpu"
            ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=os.environ.copy())
        # Return predicted mask path
        pred_mask = os.path.join(nnunet_out, f"{base}.nii.gz")
        return pred_mask

def apply_inverse_flirt(pred_mask, flirt_mat, ref_img, output_path):
    """
    Apply inverse FLIRT transform to bring mask back to original MRI space.
    Requires FSL installed and in PATH.
    """
    # Invert the matrix
    inv_mat = flirt_mat.replace(".mat", "_inv.mat")
    subprocess.run(["convert_xfm", "-inverse", "-omat", inv_mat, flirt_mat], check=True)
    # Apply inverse transform
    subprocess.run([
        "flirt", "-in", pred_mask, "-ref", ref_img,
        "-applyxfm", "-init", inv_mat, "-interp", "nearestneighbour",
        "-out", output_path
    ], check=True)
    return output_path

def main():
    # Paths
    preproc_dir = "/deneb_disk/TR2_data/nnunet_input/preprocessed_data"
    orig_dir = "/deneb_disk/TR2_data/nnunet_input/"
    mni_template = "/deneb_disk/TR2_data/ATLAS_Data/ATLAS_2/MNI152NLin2009aSym.nii.gz"
    nnunet_results = "/home/ajoshi/project2_ajoshi_1183/data/TR2/nnUNet_results"
    dataset_id = 1
    config = "3d_fullres"
    fold = 0
    checkpoint = "checkpoint_best.pth"
    out_final = os.path.join(orig_dir, "final_masks_nnunet")
    os.makedirs(out_final, exist_ok=True)

    for mni_img in sorted(glob.glob(os.path.join(preproc_dir, "*_1mm.nii.gz"))):
        base = os.path.basename(mni_img).replace("_1mm.nii.gz", "")
        flirt_mat = os.path.join(preproc_dir, f"{base}_to_mni.mat")
        orig_img = os.path.join(orig_dir, f"{base}.nii.gz")
        print(f"\nProcessing: {base}")
        # 1. Predict stroke mask in MNI space
        pred_mask = run_nnunet_predict(mni_img, preproc_dir, dataset_id, config, fold, checkpoint)
        # 2. Apply inverse transform to get mask in original MRI space
        out_mask = os.path.join(out_final, f"{base}_stroke_origspace.nii.gz")
        apply_inverse_flirt(pred_mask, flirt_mat, orig_img, out_mask)
        print(f"Saved: {out_mask}")

if __name__ == "__main__":
    main()
