import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import CamvidDataset
from small_unet import SmallUNet
from local_utils import show_images, show_image
from pretrained_model import fcn_resnet50_model

PRETRAINED = True

root_path = Path("CamVid")
train_set = CamvidDataset(root_path=root_path,split_type='train', debug=True)
train_loader = DataLoader(dataset=train_set, batch_size=1)
val_set = CamvidDataset(root_path=root_path,split_type='val', debug=True)
val_loader = DataLoader(dataset=val_set, batch_size=1)

if PRETRAINED:
    model = fcn_resnet50_model()
    load_pth = "models/model_resnet_2023-04-20-13-30-08_28.pth"
else:
    model = SmallUNet(in_channels=3, out_channels=32)
    load_pth = "models/model_unet_47.pth"

for batch_imgs, batch_masks in train_loader:
    pred = model(batch_imgs)
    if PRETRAINED:
        pred = pred['out']
    pred = torch.nn.functional.softmax(pred.squeeze(), dim=0)
    seg_map = torch.argmax(pred, dim=0).detach().cpu().numpy()
    print('-'*10)
    img = batch_imgs.squeeze()
    # plt.imshow(img.long().permute(1,2,0))
    # show_images([img.long(), batch_masks[0,:,:], seg_map])
    break
model.to('cpu')
saved_model = torch.load(load_pth,map_location=torch.device('cpu'))['model_state_dict']
# model = SmallUNet(in_channels=3, out_channels=32)
model.load_state_dict(saved_model)
# exit(0)
i = 0
samples= []
for batch_imgs, batch_masks in val_loader:
    pred = model(batch_imgs)
    if PRETRAINED:
        pred = pred['out']
    pred = torch.nn.functional.softmax(pred.squeeze(), dim=0)

    seg_map = torch.argmax(pred, dim=0).detach().cpu().numpy()
    img = batch_imgs.squeeze()
    # show_images([img.long(), batch_masks[0,:,:], seg_map])
    samples.extend([img.long(), batch_masks[0,:,:], seg_map])
    i+=1
    if i> 7:
        break

show_images(samples,nrows=len(samples) //6)
