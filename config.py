import argparse

parser = argparse.ArgumentParser(description='PyTorch 3D CNN')

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', type=int, default=500,help='Number of epochs')
parser.add_argument('--batch_size', type=int,default=4,help='Batch size')
parser.add_argument('--lr', type=float, default=0.0001,help='Learning rate')
parser.add_argument('--load', dest='load', type=str, default=False,help='Load model from a .pth file')
parser.add_argument('--scale', dest='scale', type=float, default=0.5,help='Downscaling factor of the images')
parser.add_argument('--validation', dest='val', type=float, default=10.0,help='Percent of the data that is used as validation (0-100)')
parser.add_argument('--print_freq', type=int, default=1)

parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--crop_size',default='64,128,128')
parser.add_argument('--model',type=str, default='unet')
parser.add_argument('--loss',type=str, default='CE')
parser.add_argument('--aux_loss',type=str, default='dice')
parser.add_argument('--lbd',type=float, default=20.0)


parser.add_argument('--output_dir', default='./runs/')
parser.add_argument('--train_img_folder', default='./data/imgs_crop/')
parser.add_argument('--train_mask_folder', default='./data/masks_crop/')
parser.add_argument('--test_img_folder', default='./data/test/')

args = parser.parse_args()
