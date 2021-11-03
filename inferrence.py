
import numpy as np
import torch,os,shutil
import torch.nn.functional as F
from model_zoo.unet2d import UNet
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def predict_mask(net, full_img, device, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(full_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        full_mask = output.squeeze().cpu().numpy()
        print(full_mask)
    # mask = np.argmax(full_mask > out_threshold, axis=0).astype(np.int16)
    mask = (full_mask > out_threshold).astype(np.int16)
    return mask

def Inference_single_image(net, model_ckpt, img_path, output_path='./'):   
    net.to(device=device)
    net.load_state_dict(torch.load(model_ckpt, map_location=device)['state_dict'])

    img = nib.load(img_path)
    img_data = img.get_fdata()
    img_affine = img.affine
    img_data = img_data[np.newaxis, :]
    sub_id = (img_path.split('/')[-1]).replace('T2w', 'pred')
    print(sub_id)
    mask = predict_mask(net=net,full_img=img_data,out_threshold=0.5,device=device)
    inference_path = os.path.join(output_path, sub_id)
    predict_img = nib.Nifti1Image(mask, img_affine).to_filename(inference_path)
    shutil.copyfile(img_path,os.path.join(output_path, (img_path.split('/')[-1])))

def Inference_Folder_images(net, model_ckpt, folder_path, output_path='./'):
    net.to(device=device)
    net.load_state_dict(torch.load(model_ckpt, map_location=device)['state_dict'])
    
    if not os.path.exists(output_path):
        os.makedirs(output_path + 'pred')


    for img_path in os.listdir(folder_path):
        img = Image.open((os.path.join(folder_path,img_path)))
        img = np.asarray(img)
        img = img[np.newaxis, ...]
        img = img / 255

        sub_id = str(img_path.split('.')[0])
        mask = predict_mask(net=net, full_img=img, out_threshold=0.5, device=device)
        inference_path = os.path.join(output_path+ 'pred', sub_id)
        img = Image.fromarray((mask * 255).astype(np.uint8))
        out_filename = os.path.join(output_path,  'pred/' + sub_id + '.tif')
        img.save(out_filename)

           
if __name__ == "__main__":
    model = "/data/ziyang/workspace/Machine-Learning-U-Net/runs/2DUnet_baseline_0_dice_loss_/unet_best_model.pth.tar"
    folder = "/data/ziyang/workspace/Machine-Learning-U-Net/images/train/images"
    net = UNet(n_channels=1, n_classes=1)
    Inference_Folder_images(net, model, folder,'/data/ziyang/workspace/Machine-Learning-U-Net/runs/2DUnet_baseline_0_dice_loss_/train/')

    

