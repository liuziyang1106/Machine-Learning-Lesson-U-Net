import numpy as np
import SimpleITK as sitk
from glob import glob
import os

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) 
    resampler.SetReferenceImage(itkimage)  
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  
    return itkimgResampled


image_path = './data/imgs/'
os.makedirs('./data/imgs_resize')
image_file = glob(image_path + '*.nii.gz')
for i in range(len(image_file)):
    itkimage = sitk.ReadImage(image_file[i])
    itkimgResampled = resize_image_itk(itkimage, (128, 128, 128),resamplemethod= sitk.sitkNearestNeighbor) #这里要注意：mask用最近邻插值，CT图像用线性插值
    sitk.WriteImage(itkimgResampled,'./data/imgs_resize/' + image_file[i][len(image_path):])