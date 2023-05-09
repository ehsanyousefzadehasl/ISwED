from torch import nn
from pathlib import Path
import numpy as np
from helper import get_label_mask, class_values
import torch.utils.data
import torchvision.transforms.functional
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


class CamvidDataset(torch.utils.data.Dataset):
    """
    ## Camvid Dataset
    """

    def __init__(self, root_path: Path, split_type='train', debug:bool=False, img_dim:int =512):
        """
        :param image_path: is the path to the images
        :param mask_path: is the path to the masks
        """
        # Get a dictionary of images by id
        image_path = root_path / split_type
        self.split_type=split_type
        self.images = {p.stem: p for p in image_path.iterdir()}
        # Get a dictionary of masks by id
        self.masks = {k: self.get_mask(img_p) for k, img_p in self.images.items()}
        # Image ids list
        self.ids = list(self.images.keys())
        self.debug = debug
        if self.debug:
            self.ids = self.ids[:16]

        # Transformations
        self.img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.Resize((img_dim,img_dim)),
            torchvision.transforms.Normalize(
            mean=[0.45734706, 0.43338275, 0.40058118],
            std=[0.23965294, 0.23532275, 0.2398498]
            )
        ])
        self.mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.Resize((img_dim,img_dim), interpolation=InterpolationMode.NEAREST),
        ])

    def get_mask(self,img_path:Path) ->Path:
        mask_name=img_path.with_suffix("").name + "_L.png"
        return img_path.parent.parent/f"{self.split_type}_labels/{mask_name}"

    def __getitem__(self, idx: int):
        """
        #### Get an image and its mask.

        :param idx: is index of the image
        """

        # Get image id
        id_ = self.ids[idx]
        # Load image
        img = np.array(Image.open(self.images[id_]).convert('RGB'))
        mask = np.array(Image.open(self.masks[id_]).convert('RGB'))         
        
        # rearrange dimensions
        img = np.transpose(img, (2, 0, 1))
        
        # transform the colored masks to class labels per pixel
        mask = get_label_mask(mask, class_values)

        # 
        img = torch.Tensor(img)
        mask = torch.Tensor(mask).unsqueeze(0)

        # apply transformations
        img = self.mask_transforms(img)
        mask = self.mask_transforms(mask)
        
        # rm singular dimension
        mask = mask.squeeze()

        img = img.float()
        mask = mask.long()
        return img, mask

    def __len__(self):
        """
        #### Size of the dataset
        """
        return len(self.ids)