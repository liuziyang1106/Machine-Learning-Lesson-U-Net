import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def resampleVolume(outspacing,vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0,0,0]
    inputspacing = 0
    inputsize = 0
    inputorigin = [0,0,0]
    inputdir = [0,0,0]

    #读取文件的size和spacing信息
    
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    #计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(inputsize[0]*inputspacing[0]/outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1]*inputspacing[1]/outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2]*inputspacing[2]/outspacing[2] + 0.5)

    #设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol

def resample_single_file():
    #读文件
    vol = sitk.Image(sitk.ReadImage("/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/imgs/sub-041_rec-irtk_T2w.nii.gz"))
    #重采样
    newvol = resampleVolume([0.5,0.5,0.5],vol)
    #写文件
    wriiter = sitk.ImageFileWriter()
    wriiter.SetFileName("./output2.nii.gz")
    wriiter.Execute(newvol)

def resample_folder(folder_path, target_path, spacing=[0.5,0.5,0.5]):

    for file in os.listdir(folder_path):
        #读文件
        print(file)
        vol = sitk.Image(sitk.ReadImage(os.path.join(folder_path, file)))
        #重采样
        newvol = resampleVolume(spacing,vol)
        #写文件
        wriiter = sitk.ImageFileWriter()
        wriiter.SetFileName(os.path.join(target_path, file))
        wriiter.Execute(newvol)


def center_crop(file_path, seg_path, sub_id, img_target_path, mask_target_path):
    print(file_path)
    raw_img, seg_img = sitk.Image(sitk.ReadImage(file_path)), sitk.Image(sitk.ReadImage(seg_path))
    resample_img, resample_mask = resampleVolume([0.5,0.5,0.5] ,raw_img), resampleVolume([0.5,0.5,0.5],seg_img)
    resample_img, resample_mask = sitk.GetArrayFromImage(resample_img), sitk.GetArrayFromImage(resample_mask)

    assert resample_img.shape == resample_mask.shape
    print(resample_img.shape, resample_mask.shape)

    resample_img = np.pad(resample_img,(100,100) ,mode="constant")
    resample_mask = np.pad(resample_mask,(100,100) ,mode="constant")

    assert resample_img.shape == resample_mask.shape
    print(resample_img.shape, resample_mask.shape)

    center_coord = ndimage.measurements.center_of_mass(resample_img)
    crop_size = 100
    crop_img = resample_img[int(center_coord[0])-crop_size:int(center_coord[0])+crop_size
                          ,int(center_coord[1])-crop_size:int(center_coord[1])+crop_size
                          ,int(center_coord[2])-crop_size:int(center_coord[2])+crop_size
                          ]

    crop_seg = resample_mask[int(center_coord[0])-crop_size:int(center_coord[0])+crop_size
                            ,int(center_coord[1])-crop_size:int(center_coord[1])+crop_size
                            ,int(center_coord[2])-crop_size:int(center_coord[2])+crop_size
                            ]

    assert crop_img.shape == crop_seg.shape
    print(crop_img.shape, crop_seg.shape)
    crop_img, crop_seg = sitk.GetImageFromArray(crop_img), sitk.GetImageFromArray(crop_seg)
    raw_path = os.path.join(img_target_path, 'img_resample_crop',sub_id+'_T2w.nii.gz')
    seg_save_path = os.path.join(mask_target_path, 'mask_resample_crop',sub_id+'_dseg.nii.gz')

    img_wriiter,seg_writer = sitk.ImageFileWriter(),sitk.ImageFileWriter()
    img_wriiter.SetFileName(raw_path)
    seg_writer.SetFileName(seg_save_path)
    img_wriiter.Execute(crop_img)
    seg_writer.Execute(crop_seg)
    # raw_img = nib.Nifti1Image(crop_img, raw_affine).to_filename(raw_path)
    # seg_img = nib.Nifti1Image(crop_seg, seg_affine).to_filename(seg_save_path)

def main():
    root_path = "/home/liuziyang/workspace/FeTA/data/raw_data"
    for sub_folder in os.listdir(root_path):
        data_folder = os.path.join(root_path, sub_folder, 'anat')
        for files in os.listdir(data_folder):
            if 'T2w.nii.gz' in files:
                file_path = os.path.join(data_folder, files)
            elif 'dseg.nii.gz' in files:
                seg_path = os.path.join(data_folder, files)

        center_crop(file_path, seg_path, sub_folder
        , "/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/"
        , "/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/")
main()
