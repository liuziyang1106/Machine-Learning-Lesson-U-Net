import cv2 as cv
import numpy as np
from scipy.spatial.distance import euclidean
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
from sklearn.utils.extmath import cartesian
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors._kde import KernelDensity

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
"""

class HausdorffDTLoss(nn.Module):
    '''
    Binary Hausdorff loss based on distance transform
    '''

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss,self).__init__()
        self.alpha = alpha
    
    @torch.no_grad()
    def distance_field(self, img):
        field = np.zeros_like(img)
        
        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field


    def forward(self, pred, target, debug=False) :
        assert pred.dim() == 4 or pred.dim() == 5           
        ''' Only 2D and 3D supported '''
        assert (pred.dim() == target.dim())                
        ''' Prediction and target need to be of same dimension ''' 

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance

        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss



"""
Hausdorff loss implementation based on Github reps:
https://github.com/HaipengXiong/weighted-hausdorff-loss/blob/master/object-locator/losses.py
"""

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||

    '''

    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    '''
    [Compute the Averaged Hausdorff Distance function between two inordered set
    of points (the function is symmetric).
    Note: Batchs are not supported, so squeeze your inputs first!]

    Args:
        set1 ([Array or list]): [Array/list where each row/element is an N-dimensional point.]
        set2 ([Array or list]): [Array/list where each row/element is an N-dimensional point.]
        max_ahd: [Maximum AHD possible to return if any set is empty.]. Defaults to np.inf.
    Return: 
        The Averaged Hausdorff Distance between set1 and set2.

    '''
    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim
    
    assert set1.shape[1] == set2.shape[1], \
    'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])
    
    '''
    [pairwise_distances]:
    Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns a distance matrix.
    If the input is a vector array, the distances are computed.
    If the input is a distances matrix, it is returned instead.

    This method provides a safe way to take a distance matrix as input, 
    while preserving compatibility with many other algorithms that take a vector array.

    If Y is given (default is None), then the returned matrix is the pairwise distance between the arrays from both X and Y.
    '''
    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')
    res = np.average(np.min(d2_matrix, axis=0)) + np.average(np.min(d2_matrix, axis=1))

    return res


class AveragedHaursdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """ 
        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)

        # Modified Chamfer Loss
        term1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term1 + term2
        return res



class HaursdorffLoss(nn.Module):
    def __init__(self):
        super(HaursdorffLoss, self).__init__()

    def forward(self, pred, target):
        # pred Input shape B, C, H, W D
        pred = torch.softmax(pred, dim=1)        
        pred = pred.squeeze(0)
        pred = torch.argmax(pred, dim=1)                 #B, H, W D

        
        
        print(pred.shape, target.shape)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    pred = torch.autograd.Variable(torch.rand(2,8,4,4,4)).to(device)
    target = torch.autograd.Variable(torch.rand(2,4,4,4)).to(device)
    criterion = HaursdorffLoss()
    criterion(pred, target)
