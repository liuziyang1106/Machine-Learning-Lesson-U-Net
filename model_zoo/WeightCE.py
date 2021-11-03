import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class WeightedCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropy,self).__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        target = target.long()
        num_classes = pred.size()[1]
        
        i0 = 1
        i1 = 2

        while i1 < len(pred.shape): # this is ugly but torch only allows to transpose two axes at once
            pred = pred.transpose(i0, i1)
            i0 += 1
            i1 += 1

        pred = pred.contiguous()
        pred = pred.view(-1, num_classes)

        target = target.view(-1,)
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(pred, target)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    pred = torch.autograd.Variable(torch.rand(2,8,4,4,4)).to(device)
    target = torch.autograd.Variable(torch.rand(2,4,4,4)).to(device)
    weight = torch.cuda.FloatTensor([1.14, 3.44, 5.19, 3.33,
                                     9.45, 13.28, 9.35, 17.49])
    criterion = WeightedCrossEntropy(weight)
    loss = criterion(pred, target)
    print(loss)