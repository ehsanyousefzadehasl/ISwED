from datetime import datetime
from pathlib import Path
from torchmetrics import JaccardIndex
import torch
# from lion_pytorch import Lion
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CamvidDataset
from helper import get_args, save_model_dict
from pretrained_model import fcn_resnet50_model
from small_unet import SmallUNet

args = get_args()
###### HYPERPARAMS ######
EPOCHS = args.epochs
LR = args.lr
TRANSFER_LEARNING = args.transfer_learning
BATCH_SIZE = args.batch_size
DEBUG = args.debug
print(args)
###### INIT ######
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

## data
root_path = Path("CamVid")
train_set = CamvidDataset(root_path=root_path, split_type='train', debug=DEBUG)
val_set = CamvidDataset(root_path=root_path, split_type='val', debug=DEBUG)

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE)

## model
if TRANSFER_LEARNING:
    model = fcn_resnet50_model()
else:
    model = SmallUNet(in_channels=3,out_channels=32)

criterion = nn.CrossEntropyLoss(ignore_index=255)
iou = JaccardIndex(task='multiclass',num_classes=32)
softmax = nn.Softmax()
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
# optim = Lion(model.parameters(), lr=LR, weight_decay=1e-3)

model = model.to(device)
iou = iou.to(device)

## logging

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

writer = SummaryWriter(log_dir=f"tb_logs/{'resnet' if TRANSFER_LEARNING else'unet'}/{timestamp}")
# from torchsummary import summary
# summary(model,(3,512,512))


### Train loop ###
print("start training")
for epoch in range(EPOCHS):
    model.train()
    loss = 0
    for batch_imgs, batch_masks in train_loader:
        # move to gpu if available
        batch_imgs = batch_imgs.to(device)
        batch_masks = batch_masks.to(device)

        optim.zero_grad()
        logits = model(batch_imgs)
        if TRANSFER_LEARNING:
            # needed for pretrained fcn resnet50 since model output is wrapped in a dict
            logits = logits['out'] 
        train_loss = criterion(logits, batch_masks)
        train_loss.backward()

        optim.step()
        loss += train_loss.item()

    loss = loss / len(train_loader)

    # compute validation loss
    val_loss = 0
    val_iou = 0
    with torch.no_grad():
        model.eval()
        for batch_imgs, batch_masks in val_loader:
            # move to gpu if available
            batch_imgs = batch_imgs.to(device)
            batch_masks = batch_masks.to(device)

            logits = model(batch_imgs)
            if TRANSFER_LEARNING:
                # needed for pretrained fcn resnet50 since model output is wrapped in a dict
                logits = logits['out'] 
            cur_loss = criterion(logits, batch_masks)
            cur_iou = iou(logits, batch_masks)
            val_loss += cur_loss.item()
            val_iou += cur_iou.item()
        val_loss = val_loss / len(val_loader)
        val_iou = val_iou / len(val_loader)
    # logging
    if epoch % 1 == 0:
        print("epoch : {}/{}, train_loss = {:.6f}".format(epoch + 1, EPOCHS, loss))
        print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1, EPOCHS, val_loss))
        print("epoch : {}/{}, val_iou = {:.6f}".format(epoch + 1, EPOCHS, val_iou))
    writer.add_scalar(f"{criterion}/train", loss, epoch + 1)
    writer.add_scalar(f"{criterion}/val", val_loss, epoch + 1)
    writer.add_scalar(f"{criterion}/val_iou", val_iou, epoch + 1)
    writer.add_scalar(f"{criterion}/val", val_loss, epoch + 1)
    writer.add_scalars(f"{criterion}/train&val", {
        'train': loss,
        'validation': val_loss
    }, epoch + 1)
    writer.flush()

    # save checkpoint
    if epoch >= 23: 
        save_model_dict(
            model=model,
            epoch=epoch,
            optimizer=optim,
            criterion=criterion,
            idx= f'resnet_{timestamp}' if TRANSFER_LEARNING else f'unet_{timestamp}'
        )