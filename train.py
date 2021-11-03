import tensorboardX
import os,torch,datetime,sys,logging
import numpy as np
import torch.nn as nn
from torch import optim
from model_zoo.unet3d import UNet
from inferrence import *
from model_zoo.dice_loss import DiceLoss
from model_zoo.focal_loss import FocalLoss
from model_zoo.WeightCE import WeightedCrossEntropy
from utils.weight_init import weights_init
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from config import args
from evaluation import *
torch.backends.cudnn.enabled = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_size = [int(d) for d in args.crop_size.split(',')]


def main(res):
    best_metric = 100

    model = UNet(n_channels=1, n_classes=8, trilinear=True).to(device)
    model.apply(weights_init)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    dataset = BasicDataset(args.train_img_folder, args.train_mask_folder, crop_size=img_size)
    n_val = int(len(dataset) * args.val / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True
                             ,num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False
                             ,num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.5,verbose=1)

    # Setting the loss function
    loss_func_dict = {'CE': nn.CrossEntropyLoss().to(device)
                     ,'dice':DiceLoss().to(device)
                     ,'focal':FocalLoss().to(device)
                     ,'WCE':WeightedCrossEntropy(torch.cuda.FloatTensor([1.18, 5.28, 5.49, 3.62,
                                                                         9.32, 14.92, 10.48, 17.65])).to(device)
                     ,'L1':nn.L1Loss().to(device)
                     ,'L2':nn.MSELoss().to(device)}

    criterion = loss_func_dict[args.loss]
    aux_criterion = loss_func_dict[args.aux_loss]

    saved_metrics, saved_epos = [], []
    writer = tensorboardX.SummaryWriter(args.output_dir)

    for epoch in range(args.epochs):
        train_loss, train_aux_loss = train(train_loader, model=model, criterion=criterion, aux_criterion = aux_criterion
                                          ,optimizer = optimizer, epoch = epoch, device = device)
        val_loss, val_aux_loss = valiation(val_loader=valid_loader, model=model, criterion=criterion, aux_criterion=aux_criterion
                                          ,device=device)
        scheduler.step()

        # ===========  write in tensorboard scaler =========== #
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/aux_loss', train_aux_loss, epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)
        writer.add_scalar('Val/aux_loss', val_aux_loss, epoch)


        valid_metric = val_aux_loss
        is_best = False
        if valid_metric < best_metric:
            is_best = True
            best_metric = min(valid_metric, best_metric)
                
            saved_metrics.append(valid_metric)
            saved_epos.append(epoch)
            print('=======>   Best at epoch %d, valid Dice Loss %f\n' % (epoch, best_metric))

        save_checkpoint({'epoch': epoch
                        ,'state_dict': model.state_dict()}
                        , is_best, args.output_dir, model_name=args.model)
    # =========== write traning and validation log =========== #
    os.system('echo " ================================== "')
    os.system('echo " ==== TRAIN MAE mtc:{:.5f}" >> {}'.format(train_loss, res))
    
    print('Epo - Mtc')
    mtc_epo = dict(zip(saved_metrics, saved_epos))
    rank_mtc = sorted(mtc_epo.keys(), reverse=False)
    try:
        for i in range(5):
            print('{:03} {:.3f}'.format(mtc_epo[rank_mtc[i]]
                                       ,rank_mtc[i]))
            os.system('echo "epo:{:03} mtc:{:.3f}" >> {}'.format(mtc_epo[rank_mtc[i]],rank_mtc[i],res))
    except:
        pass
    
    # ===========  clean up ===========  #
    torch.cuda.empty_cache()
    
    model_ckpt = os.path.join(args.output_dir, args.model+'_best_model.pth.tar')
    Inference_Folder_images(model, model_ckpt, args.train_img_folder, os.path.join(args.output_dir, 'pred_train_mask/'))
    Inference_Folder_images(model, model_ckpt, args.test_img_folder , os.path.join(args.output_dir, 'pred_test_mask/'))

    evl_test = SegEval(os.path.join(args.output_dir, 'pred_test_mask/pred')
                      ,os.path.join(args.output_dir, 'pred_test_mask/mask'))
    evl_test.evaluation_by_folder(["dice", "acc", "hausdorff", "volume similarity", "sensitivity", "precision"])
    evl_test.export_eval_results(args.output_dir,'test_results.xlsx')

    evl_train = SegEval(os.path.join(args.output_dir, 'pred_train_mask/pred')
                       ,os.path.join(args.output_dir, 'pred_train_mask/mask'))
    evl_train.evaluation_by_folder(["dice", "acc", "hausdorff", "volume similarity", "sensitivity", "precision"])
    evl_train.export_eval_results(args.output_dir, 'train_results.xlsx')
    return 0

def train(train_loader, model, criterion, aux_criterion, optimizer, epoch, device):

    Epoch_loss1 = AverageMeter()
    AUX_loss = AverageMeter()
    Total_loss = AverageMeter()

    for i, batch in enumerate(train_loader):
        imgs = batch['image']
        true_masks = batch['mask']
        assert imgs.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
        optimizer.zero_grad()

        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if model.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)
        masks_pred = model(imgs)

        # # Remove the axis
        true_masks = torch.squeeze(true_masks, dim=1)
        loss1 = criterion(masks_pred, true_masks)
        aux_loss = aux_criterion(masks_pred, true_masks)
        loss = loss1 + args.lbd * aux_loss
        
        Epoch_loss1.update(loss1, imgs.size(0))
        AUX_loss.update(aux_loss, imgs.size(0))
        Total_loss.update(loss, imgs.size(0))
        if i % args.print_freq == 0:
            print('Epoch: [{0} / {1}]   [step {2}/{3}] \t'
                'Tot_Loss {tot_loss.val:.4f} ({tot_loss.avg:.4f})  \t'
                'Main_Loss {loss.val:.4f} ({loss.avg:.4f})  \t'
                'Aux_Loss {aux_loss.val:.4f} ({aux_loss.avg:.4f})  \t'.format
                (epoch, args.epochs, i, len(train_loader), tot_loss=Total_loss,loss=Epoch_loss1, aux_loss=AUX_loss))
    
        loss.backward()
        optimizer.step()

    return Epoch_loss1.avg, AUX_loss.avg

def valiation(val_loader, model, criterion, aux_criterion, device):
    Epoch_loss = AverageMeter()
    AUX_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs = batch['image']
            true_masks = batch['mask']
            assert imgs.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            masks_pred = model(imgs)

            # Remove the axis
            true_masks = torch.squeeze(true_masks, dim=1)
            loss = criterion(masks_pred, true_masks)
            aux_loss = aux_criterion(masks_pred, true_masks)

            Epoch_loss.update(loss, imgs.size(0))
            AUX_loss.update(aux_loss, imgs.size(0))

        print('Valid: [steps {0}], Main_Loss {loss.avg:.4f} Aux_Loss {Aux_loss.avg:.4f}'.format(
               len(val_loader), loss=Epoch_loss, Aux_loss=AUX_loss))
    return Epoch_loss.avg, AUX_loss.avg

def save_checkpoint(state, is_best, out_dir, model_name):
    checkpoint_path = out_dir+model_name+'_checkpoint.pth.tar'
    best_model_path = out_dir+model_name+'_best_model.pth.tar'
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)
        print("=======>   This is the best model !!! It has been saved!!!!!!\n\n")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    print(args)
    res = os.path.join(args.output_dir, 'result.txt')
    if os.path.isdir(args.output_dir): 
        if input("### output_dir exists, rm? ###") == 'y':
            os.system('rm -rf {}'.format(args.output_dir))

    # =========== set train folder =========== #
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print('=> training from scratch.\n')
    os.system('echo "train {}" >> {}'.format(datetime.datetime.now(), res))
    main(res)
