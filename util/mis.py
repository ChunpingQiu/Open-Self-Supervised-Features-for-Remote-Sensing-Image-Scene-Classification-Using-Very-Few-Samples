
from PIL import Image
import  rasterio
import numpy as np

from torchgeo.transforms import AugmentationSequential
import torchvision.transforms as transforms

import torch


def processMB(img, bands, augmentations):
        
    with rasterio.open(img) as dataset:
        array = dataset.read()
    # print(array.dtype)
    # print(np.shape(array))

    # print(, np.shape(array[[4,3,2],:,:]))
    x=array[bands,:,:]
    # print(np.shape(x))
    # x=np.moveaxis(x, 0, -1)#.shape
    # print(np.shape(x))

    # new_img = Image.fromarray(x)
    # print(new_img)


    # x = x*0.0001
    min_16bit = np.min(x.flatten())
    max_16bit = np.max(x.flatten())
    # x = np.array(np.rint(255 * ((x - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    x = np.array(np.float16( ((x - min_16bit) / (max_16bit - min_16bit))), dtype=np.float16)
    
    
    imageTmp = torch.from_numpy(x)
    sample = {"image": imageTmp}
    # sample: Dict[str, Tensor] = {"image": imageTmp}
    sample = augmentations(sample)
    imageTmp = sample["image"]

    imageTmp = imageTmp[np.newaxis, ...]
    
    return imageTmp