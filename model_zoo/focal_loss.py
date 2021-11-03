import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

class FocalLoss(nn.Module):
    '''
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    '''

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2.0, balance_index=0, smooth=1e-5, size_average=True):

        '''
        [summary]

        Args:
            apply_nonlin ([type], optional): [description]. Defaults to None.
            alpha ([Tensor], optional): [3D or 4D the scalar factor for this criterion]. Defaults to None.
            gamma (float, optional): [gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                                      focus on hard misclassified example]. Defaults to 2.
            balance_index (int, optional): [balance class index, should be specific when alpha is float]. Defaults to 0.
            smooth (float, optional): [smooth value when cross entropy]. Defaults to 1e-5.
            size_average (bool, optional): [By default, the losses are averaged over each loss element in the batch.]. Defaults to True.
        '''
        super(FocalLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

        
    def forward(self, logit, target):
        # if self.apply_nonlin is not None:
        #     logit = self.apply_nonlin(logit)
        logit = F.softmax(logit,dim=1)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N, C, d1, d2 -> N, C, m (m=d1*d2*dn...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0,2,1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = torch.squeeze(target, -1)
        target = target.view(-1, 1)
        
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class,1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1-pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()

        else:
            loss = loss.sum()

        return loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    pred = torch.autograd.Variable(torch.rand(2,8,4,4,4)).to(device)
    target = torch.autograd.Variable(torch.rand(2,4,4,4)).to(device)
    criterion = FocalLoss()
    loss = criterion(pred, target)
    print(loss)
