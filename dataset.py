import numpy as np
import torch,random
from glob import glob
from os import listdir
from scipy import ndimage
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    return data

def white0(image, threshold=1):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    image_h = image * mask
    image_l = image * (1 - mask)

    mean = np.sum(image_h) / np.sum(mask)
    std = np.sqrt(np.sum(np.abs(image_h - mean)**2) / np.sum(mask))

    if std > 0:
        ret = (image_h - mean) / std + image_l
    else:
        ret = image * 0.
    return ret

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_suffix='_T2w', mask_suffix='_dseg',crop_size=(64,64,64)):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.ids = [file.replace('_T2w', '').split('.')[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.crop_size = crop_size

    @staticmethod
    def randomCrop(img, mask, width, height, depth):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]

        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        z = random.randint(0, img.shape[2] - depth)
        img = img[y:y+height, x:x+width, z:z+depth]
        mask = mask[y:y+height, x:x+width, z:z+depth]
        return img, mask

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + self.img_suffix + '.*')
        # print(self.masks_dir + idx + self.mask_suffix + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = nii_loader(mask_file[0])
        img = nii_loader(img_file[0])
        # img = white0(img)

        img, mask = self.randomCrop(img, mask, self.crop_size[0], self.crop_size[1], self.crop_size[2])
        
        img = img[np.newaxis, :]
        mask = mask[np.newaxis, :]

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'id': idx
        }

if __name__ == "__main__":

    dir_img = './data/imgs_crop/'
    dir_mask = './data/masks_crop/'
    dir_checkpoint = 'checkpoints/'
    val_percent = 0.2

    dataset = BasicDataset(dir_img, dir_mask)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    for i, batch in enumerate(train_loader):
        imgs = batch['image']
        true_masks = batch['mask']
        print(i, imgs.shape, true_masks.shape)
        # # print(true_masks.max())

        # if true_masks.max() != 7:
        #     print(batch['id'])  