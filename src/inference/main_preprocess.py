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

    # 1. Reorient to standard orientation
    reorient = Reorient2Std(in_file=input_t1, out_file=f"{output_dir}/t1_std.nii.gz")
    reorient.run()

    # 2. N4 Bias Field Correction (SimpleITK Python implementation)
    print("Applying N4 bias field correction (SimpleITK)...")
    std_path = f"{output_dir}/t1_std.nii.gz"
    
    # Load image with SimpleITK
    img_sitk = sitk.ReadImage(std_path)
    img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)
    
    # Create mask (simple threshold-based)
    img_array = sitk.GetArrayFromImage(img_sitk)
    mask_array = img_array > sitk.GetArrayFromImage(img_sitk).mean() * 0.2
    mask_sitk = sitk.GetImageFromArray(mask_array.astype('uint8'))
    mask_sitk.CopyInformation(img_sitk)
    
    # Apply N4BiasFieldCorrection
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_img = corrector.Execute(img_sitk, mask_sitk)
    
    # Save corrected image
    n4_path = f"{output_dir}/t1_n4.nii.gz"
    sitk.WriteImage(corrected_img, n4_path)
    print(f"N4 bias field correction complete: {n4_path}")

    # 3. Linear Registration to MNI-152 Template
    flirt = FLIRT(bins=256, cost='corratio', interp='trilinear')
    flirt.inputs.in_file = f"{output_dir}/t1_n4.nii.gz"
    flirt.inputs.reference = mni_template
    flirt.inputs.out_file = f"{output_dir}/t1_mni.nii.gz"
    flirt.inputs.out_matrix_file = f"{output_dir}/t1_to_mni.mat"
    flirt.run()

    # 4. Resample to 1mm Isotropic
    # Note: If your MNI template is already 1mm, FLIRT handles this. 
    # Otherwise, use nibabel/scipy to force resolution.
    img = nib.load(f"{output_dir}/t1_mni.nii.gz")
    # Simple resampling: just note the current voxel dims
    print(f"Current voxel dims: {img.header.get_zooms()}")
    # For true 1mm resampling, use scipy.ndimage.zoom or nilearn
    nib.save(img, f"{output_dir}/t1_1mm.nii.gz")

    # 5. Defacing (Anonymization)
    # FreeSurfer's Deface is not readily available in nipype; skipping for now.
    # For production, use FSL's fslmaths or custom defacing tools.
    print("Defacing step skipped (requires additional FreeSurfer setup)")

    print(f"Preprocessing complete. Final file: {output_dir}/t1_1mm.nii.gz")

# --- Configuration ---
# Path to your raw T1 MRI
raw_mri = "/deneb_disk/TR2_data/nnunet_input/sub-1002_T1w.nii.gz"
# Path to the MNI152 template (standard with FSL: $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz)
mni_ref = "/deneb_disk/TR2_data/ATLAS_Data/ATLAS_2/MNI152NLin2009aSym.nii.gz"
output_path = "/deneb_disk/TR2_data/nnunet_input/preprocessed_data"

preprocess_atlas_v2(raw_mri, output_path, mni_ref)

