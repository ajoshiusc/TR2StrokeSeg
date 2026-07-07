import os
import glob
import nibabel as nib
import numpy as np

strong_dir = "/deneb_disk/STRONG_neuroimages_to_share/"
subjects = sorted(glob.glob(os.path.join(strong_dir, "sub-*")))

sizes = []
for sub in subjects[:50]: # just test first 50 to find a good mix
    mask_path = glob.glob(os.path.join(sub, "*_mask_mni_1mm.nii.gz"))
    if mask_path:
        mask = nib.load(mask_path[0]).get_fdata()
        vol = np.sum(mask > 0)
        sizes.append((os.path.basename(sub), vol, mask_path[0]))

sizes.sort(key=lambda x: x[1])
print("Smallest:", sizes[0])
print("Median:", sizes[len(sizes)//2])
print("Largest:", sizes[-1])
