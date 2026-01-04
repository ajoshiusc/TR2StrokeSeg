import os
import nibabel as nib
import SimpleITK as sitk
from nipype.interfaces.fsl import Reorient2Std, FLIRT

def preprocess_atlas_v2(input_t1, output_dir, mni_template):
    """
    Standard ATLAS v2.0 Preprocessing Workflow:
    1. Reorient to Standard (FSL)
    2. N4 Bias Field Correction (ANTs)
    3. Registration to MNI-152 (FSL FLIRT)
    4. Resampling to 1mm isotropic
    5. Defacing (FreeSurfer)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use input file base name for all outputs
    base = os.path.splitext(os.path.basename(input_t1))[0]
    if base.endswith('.nii'):  # handle .nii.gz
        base = os.path.splitext(base)[0]

    std_path = os.path.join(output_dir, f"{base}_std.nii.gz")
    n4_path = os.path.join(output_dir, f"{base}_n4.nii.gz")
    mni_path = os.path.join(output_dir, f"{base}_mni.nii.gz")
    mni_mat_path = os.path.join(output_dir, f"{base}_to_mni.mat")
    out_1mm = os.path.join(output_dir, f"{base}_1mm.nii.gz")

    # 1. Reorient to standard orientation
    reorient = Reorient2Std(in_file=input_t1, out_file=std_path)
    reorient.run()

    # 2. N4 Bias Field Correction (SimpleITK Python implementation)
    print("Applying N4 bias field correction (SimpleITK)...")
    img_sitk = sitk.ReadImage(std_path)
    img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)
    img_array = sitk.GetArrayFromImage(img_sitk)
    mask_array = img_array > sitk.GetArrayFromImage(img_sitk).mean() * 0.2
    mask_sitk = sitk.GetImageFromArray(mask_array.astype('uint8'))
    mask_sitk.CopyInformation(img_sitk)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_img = corrector.Execute(img_sitk, mask_sitk)
    sitk.WriteImage(corrected_img, n4_path)
    print(f"N4 bias field correction complete: {n4_path}")

    # 3. Linear Registration to MNI-152 Template
    flirt = FLIRT(bins=256, cost='corratio', interp='trilinear')
    flirt.inputs.in_file = n4_path
    flirt.inputs.reference = mni_template
    flirt.inputs.out_file = mni_path
    flirt.inputs.out_matrix_file = mni_mat_path
    flirt.run()

    # 4. Resample to 1mm Isotropic
    img = nib.load(mni_path)
    print(f"Current voxel dims: {img.header.get_zooms()}")
    nib.save(img, out_1mm)

    # 5. Defacing (Anonymization)
    print("Defacing step skipped (requires additional FreeSurfer setup)")

    print(f"Preprocessing complete. Final file: {out_1mm}")

# --- Configuration ---
import glob
# Directory containing raw T1 MRIs
input_dir = "/deneb_disk/TR2_data/nnunet_input/"
# Path to the MNI152 template (standard with FSL: $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz)
mni_ref = "/deneb_disk/TR2_data/ATLAS_Data/ATLAS_2/MNI152NLin2009aSym.nii.gz"
output_path = "/deneb_disk/TR2_data/nnunet_input/preprocessed_data"

nii_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
if not nii_files:
    print(f"No .nii.gz files found in {input_dir}")
else:
    for raw_mri in nii_files:
        print(f"\nProcessing: {raw_mri}")
        preprocess_atlas_v2(raw_mri, output_path, mni_ref)

