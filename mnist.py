import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class MNIST(Dataset):
    def __init__(self,split="train",transform=None):
        self.datapath="./data/"
        self.split=split
        self.transform=transform
        with open("./data/"+self.split+"_label.txt","r") as f:
            self.labels=f.readlines()
        f.close()
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        image=Image.open(self.datapath+self.split+"/"+str(idx)+".png")
        image=torch.from_numpy(np.asarray(image)).unsqueeze(0).float()
        image= self.transform(image)
        label=int(self.labels[idx][0])
        return image,label
