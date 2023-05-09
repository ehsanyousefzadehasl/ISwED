import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import cv2

#import tensorflow
from tensorflow.keras.utils import get_source_inputs

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical ,Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout,LeakyReLU
from tensorflow.keras.optimizers import Adadelta, Nadam ,Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard

import os
import tensorflow.keras as keras
os.environ["SM_FRAMEWORK"] = "tf.keras"

from glob import glob
from pathlib import Path
import shutil
from random import sample, choice
import segmentation_models as sm


# ********** Snippet for limiting the GPU memory allocation by Tensorflow ********
# Ref: https://www.tensorflow.org/guide/gpu
import tensorflow
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tensorflow.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
# ********************************************************************************


keras.backend.set_image_data_format('channels_last')

dataset_path = Path("../../../CamVid")

def tree(directory):
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')

train_imgs = list((dataset_path / "train").glob("*.png"))
train_labels = list((dataset_path / "train_labels").glob("*.png"))
val_imgs = list((dataset_path / "val").glob("*.png"))
val_labels = list((dataset_path / "val_labels").glob("*.png"))
test_imgs = list((dataset_path / "test").glob("*.png"))
test_labels = list((dataset_path / "test_labels").glob("*.png"))

(len(train_imgs),len(train_labels)), (len(val_imgs),len(val_labels)) , (len(test_imgs),len(test_labels))

img_size = 512

assert len(train_imgs) == len(train_labels), "No of Train images and label mismatch"
assert len(val_imgs) == len(val_labels), "No of Train images and label mismatch"
assert len(test_imgs) == len(test_labels), "No of Train images and label mismatch"

sorted(train_imgs), sorted(train_labels), sorted(val_imgs), sorted(val_labels), sorted(test_imgs), sorted(test_labels);

for im in train_imgs:
    assert dataset_path / "train_labels" / (im.stem +"_L.png") in train_labels , "{im} not there in label folder"
for im in val_imgs:
    assert dataset_path / "val_labels" / (im.stem +"_L.png") in val_labels , "{im} not there in label folder"
for im in test_imgs:
    assert dataset_path / "test_labels" / (im.stem +"_L.png") in test_labels , "{im} not there in label folder"

def make_pair(img,label,dataset):
    pairs = []
    for im in img:
        pairs.append((im , dataset / label / (im.stem +"_L.png")))

    return pairs


train_pair = make_pair(train_imgs, "train_labels", dataset_path)
val_pair = make_pair(val_imgs, "val_labels", dataset_path)
test_pair = make_pair(test_imgs, "test_labels", dataset_path)

temp = choice(train_pair)
img = img_to_array(load_img(temp[0], target_size=(img_size,img_size)))
mask = img_to_array(load_img(temp[1], target_size = (img_size,img_size)))

class_map_df = pd.read_csv(dataset_path / "class_dict.csv")
     
class_map = []
for index,item in class_map_df.iterrows():
    class_map.append(np.array([item['r'], item['g'], item['b']]))
    
len(class_map)


def assert_map_range(mask,class_map):
    mask = mask.astype("uint8")
    for j in range(img_size):
        for k in range(img_size):
            assert mask[j][k] in class_map , tuple(mask[j][k])

def form_2D_label(mask,class_map):
    mask = mask.astype("uint8")
    label = np.zeros(mask.shape[:2],dtype= np.uint8)

    for i, rgb in enumerate(class_map):
        label[(mask == rgb).all(axis=2)] = i

    return label


lab = form_2D_label(mask,class_map)
np.unique(lab,return_counts=True)

class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, pair, class_map, batch_size=16, dim=(512,512,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.pair = pair
        self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()

        # Generate data
        for i in list_IDs_temp:
            # Store sample
            img = load_img(self.pair[i][0] ,target_size=self.dim)
            img = img_to_array(img)/255.
            batch_imgs.append(img)

            label = load_img(self.pair[i][1],target_size=self.dim)
            label = img_to_array(label)
            label = form_2D_label(label,self.class_map)
            label = to_categorical(label , num_classes = 32)
            batch_labels.append(label)

        return np.array(batch_imgs) ,np.array(batch_labels)

train_generator = DataGenerator(train_pair+test_pair,class_map,batch_size=4, dim=(img_size,img_size,3) ,shuffle=True)
train_steps = train_generator.__len__()
print(train_steps)

dX,y = train_generator.__getitem__(1)
print(y.shape)
     
val_generator = DataGenerator(val_pair, class_map, batch_size=4, dim=(img_size,img_size,3) ,shuffle=True)
val_steps = val_generator.__len__()
print(val_steps)


# model

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


input_img = Input(shape=(512, 512, 3),name='image_input')
c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (input_img)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (c5)
p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (p5)
c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (c6)
p6 = MaxPooling2D(pool_size=(2, 2)) (c6)
p6 = Dropout(rate=0.5) (p6)

c7 = Conv2D(1024, (3, 3), activation='relu', padding='same') (p6)
c7 = Conv2D(1024, (3, 3), activation='relu', padding='same') (c7)
c7 = Dropout(rate=0.5) (c7)

u8 = upsample_conv(512, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c6])
c8 = Conv2D(512, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(512, (3, 3), activation='relu', padding='same') (c8)
c8 = Dropout(rate=0.5) (c8)

u9 = upsample_conv(256, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c5])
c9 = Conv2D(256, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(256, (3, 3), activation='relu', padding='same') (c9)
c9 = Dropout(rate=0.5) (c9)

u10 = upsample_conv(128, (3, 3), strides=(2, 2), padding='same') (c9)
u10 = concatenate([u10, c4])
c10 = Conv2D(128, (3, 3), activation='relu', padding='same') (u10)
c10 = Conv2D(128, (3, 3), activation='relu', padding='same') (c10)

u11 = upsample_conv(64, (2, 2), strides=(2, 2), padding='same') (c10)
u11 = concatenate([u11, c3])
c11 = Conv2D(64, (3, 3), activation='relu', padding='same') (u11)
c11 = Conv2D(64, (3, 3), activation='relu', padding='same') (c11)

u12 = upsample_conv(32, (2, 2), strides=(2, 2), padding='same') (c11)
u12 = concatenate([u12, c2])
c12 = Conv2D(32, (3, 3), activation='relu', padding='same') (u12)
c12 = Conv2D(32, (3, 3), activation='relu', padding='same') (c12)

u13 = upsample_conv(16, (2, 2), strides=(2, 2), padding='same') (c12)
u13 = concatenate([u13, c1], axis=3)
c13 = Conv2D(16, (3, 3), activation='relu', padding='same') (u13)
c13 = Conv2D(16, (3, 3), activation='relu', padding='same') (c13)

d = Conv2D(32, (1, 1), activation='softmax') (c13)

##

iou = sm.metrics.IOUScore(threshold=0.5)
seg_model = Model(inputs=[input_img], outputs=[d])
seg_model.summary()

seg_model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=['accuracy',iou])

mc = ModelCheckpoint(mode='max', filepath='weights/Unet_Relu.h5', monitor='val_accuracy',save_best_only='True', save_weights_only='True', verbose=1)
# es = EarlyStopping(mode='max', monitor='val_accuracy', patience=10, verbose=1)
tb = TensorBoard(log_dir="logs/", histogram_freq=0, write_graph=True, write_images=False)
# rl = ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=10,verbose=1,mode="max",min_lr=0.0001)
cv = CSVLogger("logs/log.csv" , append=True , separator=',')
     

results = seg_model.fit(train_generator , steps_per_epoch=train_steps ,epochs=100,
                              validation_data=val_generator,validation_steps=val_steps,callbacks=[mc,tb,cv])

# visualization
from matplotlib import pyplot as plt
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(results.history['iou_score'])
plt.plot(results.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("performance.jpg")
