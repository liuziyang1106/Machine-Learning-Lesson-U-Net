import numpy
import nibabel as nib
import torch,os
import torch.nn.functional as F 


path = 'data/data_2.1/Seg/'
P = []
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    a = nib.load(file_path).get_fdata()
    unique, counts = numpy.unique(a, return_counts=True)
    prec = (180*180*180)/counts
    prec = numpy.sqrt(prec)
    P.append(prec)
    print(prec)
    # print(dict(zip(unique, prec)))

P = numpy.array(P)
P = numpy.average(P, axis=0)

print(P)