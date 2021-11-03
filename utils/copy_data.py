import os
from shutil import copyfile
import glob

img_paths = glob.glob("../feta_2.0/*/anat/*_T2w.nii.gz")
mask_paths = glob.glob("../feta_2.0/*/anat/*_dseg.nii.gz")
target_dir = "./data/"

for img_path in img_paths:
    filename = os.path.basename(img_path)
    target_path = os.path.join(target_dir, 'imgs', filename)
    copyfile(img_path, target_path)
    print(f"copy image from {img_path} to {target_path}")

for mask_path in mask_paths:
    filename = os.path.basename(mask_path)
    target_path = os.path.join(target_dir, 'masks', filename)
    copyfile(mask_path, target_path)
    print(f"copy image from {mask_path} to {target_path}")
