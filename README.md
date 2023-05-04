# Image Segmentation with Encoder Decoders
This repository contains the documentation and source code for experiments with encoder decoder models for image segmentation on different datasets.

## Background
Computer vision is a sub-field in computer science field aiming at gathering, processing, analyzing, and understanding digital images and extracting information that can be used by numerical decision-making processes.

### Definition
The process of partitioning a digital image into multiple image segments (image regions, image objects), which indeed are sets of pixels. In other words, image segmentation can be viewed as pixel labeling too.

### Goal
Simplify the representation of a digital image into something easier to understand and analyze. An image is a grid of pixels.

### Domain and Applications
- Object detection tasks
    - Face detection
    - Pedestrain detection
    - Locating specific objects in satelite images
- Recognition tasks
    - Face recognition
    - Fingerprint recognition
- Object localization
- Traffic control systems
- Video survelliance systems

### Different kinds of image segmentation
- Semantic segmentation (e.g., person and background)
- Instance segmentation (e.g. each person will be identified individually)

### Traditional Computer Vision Approaches
- Thresholding method (Region-based Segmentation) ==> (changing a grayscale image into a binary image based a threshold)

- Edge Detection method
    - Using weight matrices (filters) and convoluting them with images

- Clustering method: e.g., K-means clustering

### DL-based Appoaches
- Fully Convolutional Networks (FCNs)

## Evaluation Metrics
## Encoder-Decoder based DL Models

## State of the art approach
|TODO andres: Please add your summary of your research here on the state-of-the-art practice on image segmentation with encoder-decoder mechanisms
- https://reader.elsevier.com/reader/sd/pii/S0031320322007075?token=83FCA21D1027C3BFBF95656B895BEF0A262DF1328A65F90845CA8D0D34707028CBA43303B74BCF9E3624B7F3937DE7AB&originRegion=eu-west-1&originCreation=20230504124206

## Image Segmentation Datasets
The datasets for this task:
1. The Cambridge driving labeled Video databases (CamVids)
2. The Cityscapes Dataset
3. PASCAL Visual Object Classes (PASCAL VOC)
4. Common Objects in COntext — Coco Dataset
