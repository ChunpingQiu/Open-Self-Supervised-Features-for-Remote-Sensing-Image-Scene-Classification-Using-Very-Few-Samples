from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os


from util.mis import processMB

device = "cuda" if torch.cuda.is_available() else "cpu"

#based on data list in a txt file, which is prepared befrehand
class Dataloader_path_list(Dataset):
    def __init__(self, root, dataset, idx_class, classname, preprocess):

        self.folder = root+dataset
        self.classname = classname
        self.idx_class = idx_class
        self.preprocess = preprocess

        f_query=open(root+"/experiment_split/"+dataset+"_query_q9_"+str(idx_class)+"_"+classname+".txt","r")
        self.images_path_list = f_query.read().split('\n')


        # print(len(self.images_path_list))

    def __getitem__(self, index):

        img = self.folder+"/"+self.classname+"/"+self.images_path_list[index]
        image = self.preprocess(Image.open(img))#.unsqueeze(0)#.to(device)

        target = self.idx_class

        return image, target

    def __len__(self):
        return len(self.images_path_list)
    
class Dataloader_path_list_MB(Dataset):
    def __init__(self, root, dataset, idx_class, classname, augmentations, bands):

        self.folder = root+dataset
        self.classname = classname
        self.idx_class = idx_class
        self.preprocess = augmentations
        self.bands = bands

        f_query=open(root+"/experiment_split/"+dataset+"_query_q9_"+str(idx_class)+"_"+classname+".txt","r")
        self.images_path_list = f_query.read().split('\n')
        
        # print(len(self.images_path_list))

    def __getitem__(self, index):

        img = self.folder+"/"+self.classname+"/"+self.images_path_list[index]
        # image = self.preprocess(Image.open(img))#.unsqueeze(0)#.to(device)
        
        image = processMB(img, self.bands, self.preprocess).squeeze(0)#.to(device)

        target = self.idx_class

        return image, target

    def __len__(self):
        return len(self.images_path_list)