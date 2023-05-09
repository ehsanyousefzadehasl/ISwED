import numpy as np
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import math

label_colors_list = [
        (64, 128, 64), # animal
        (192, 0, 128), # archway
        (0, 128, 192), # bicyclist
        (0, 128, 64), #bridge
        (128, 0, 0), # building
        (64, 0, 128), #car
        (64, 0, 192), # car luggage pram...???...
        (192, 128, 64), # child
        (192, 192, 128), # column pole
        (64, 64, 128), # fence
        (128, 0, 192), # lane marking driving
        (192, 0, 64), # lane maring non driving
        (128, 128, 64), # misc text
        (192, 0, 192), # motor cycle scooter
        (128, 64, 64), # other moving
        (64, 192, 128), # parking block
        (64, 64, 0), # pedestrian
        (128, 64, 128), # road
        (128, 128, 192), # road shoulder
        (0, 0, 192), # sidewalk
        (192, 128, 128), # sign symbol
        (128, 128, 128), # sky
        (64, 128, 192), # suv pickup truck
        (0, 0, 64), # traffic cone
        (0, 64, 64), # traffic light
        (192, 64, 128), # train
        (128, 128, 0), # tree
        (192, 128, 192), # truck/bus
        (64, 0, 64), # tunnel
        (192, 192, 0), # vegetation misc.
        (0, 0, 0),  # 0=background/void
        (64, 192, 0), # wall
    ]

# all the classes that are present in the dataset
ALL_CLASSES = ['animal', 'archway', 'bicyclist', 'bridge', 'building', 'car', 
        'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve', 
        'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving', 
        'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk', 
        'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight', 
        'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
        'wall']

"""
This (`class_values`) assigns a specific class label to the each of the classes.
For example, `animal=0`, `archway=1`, and so on.
"""
class_values = [ALL_CLASSES.index(cls.lower()) for cls in ALL_CLASSES]

def get_label_mask(mask, class_values): 
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def save_model_dict(model, epoch, optimizer, criterion, idx):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion
            },
            f"checkpoints/model_{idx}_{epoch}.pth")

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--transfer_learning', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args