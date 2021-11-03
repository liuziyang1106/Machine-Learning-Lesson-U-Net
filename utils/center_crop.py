import os, shutil
import nibabel as nib
import numpy as np
from scipy import ndimage

def center_crop(file_path, seg_path, sub_id, img_target_path, mask_target_path):
    raw_img, seg_img = nib.load(file_path), nib.load(seg_path)
    raw_affine, seg_affine = raw_img.get_affine(), seg_img.get_affine()
    img_data, seg_data = raw_img.get_fdata(), seg_img.get_fdata()
    padding_img = np.pad(img_data,(100,100) ,mode="constant")
    padding_seg_img = np.pad(seg_data,(100,100) ,mode="constant")

    print(padding_img.shape)
    center_coord = ndimage.measurements.center_of_mass(padding_img)
    crop_size = 90
    crop_img = padding_img[int(center_coord[0])-crop_size:int(center_coord[0])+crop_size
                          ,int(center_coord[1])-crop_size:int(center_coord[1])+crop_size
                          ,int(center_coord[2])-crop_size:int(center_coord[2])+crop_size
                          ]

    crop_seg = padding_seg_img[int(center_coord[0])-crop_size:int(center_coord[0])+crop_size
                              ,int(center_coord[1])-crop_size:int(center_coord[1])+crop_size
                              ,int(center_coord[2])-crop_size:int(center_coord[2])+crop_size
                              ]
    print(crop_img.shape, crop_seg.shape)
    raw_path = os.path.join(img_target_path, 'T2',sub_id+'_T2w.nii.gz')
    seg_save_path = os.path.join(mask_target_path, 'Seg',sub_id+'_dseg.nii.gz')
    raw_img = nib.Nifti1Image(crop_img, raw_affine).to_filename(raw_path)
    seg_img = nib.Nifti1Image(crop_seg, seg_affine).to_filename(seg_save_path)

def main():
    root_path = "/data/ziyang/workspace/FeTA/data/raw_data_2.1"
    for sub_folder in os.listdir(root_path):
        data_folder = os.path.join(root_path, sub_folder, 'anat')
        for files in os.listdir(data_folder):
            if 'T2w.nii.gz' in files:
                file_path = os.path.join(data_folder, files)
            elif 'dseg.nii.gz' in files:
                seg_path = os.path.join(data_folder, files)

        center_crop(file_path, seg_path, sub_folder
                  , "/data/ziyang/workspace/FeTA/data/Crop_Data_2.1/"
                  , "/data/ziyang/workspace/FeTA/data/Crop_Data_2.1/")
main()