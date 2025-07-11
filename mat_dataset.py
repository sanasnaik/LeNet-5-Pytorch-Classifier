import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io
import numpy as np
from PIL import Image
import sys
# root_dir = os.path.join(os.path.dirname(__file__), "training_batches/1.mat")
# mat = scipy.io.loadmat(root_dir)
# print(mat.keys())

# sys.exit()

class MATDataset(Dataset):
    def __init__(self, mat_dir):
        self.files = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat')])
        self.mat_dir = mat_dir

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat_path = os.path.join(self.mat_dir, self.files[idx])
        mat = scipy.io.loadmat(mat_path)

       
        raw = mat['affNISTdata'][0, 0]

        # Extract image and label
        img_flat = raw[2][:, idx]             # shape: (1600,)
        img = img_flat.reshape(40, 40)        # shape: (40, 40)
        label_vector = raw[4][:, idx]         # shape: (10,)
        label = int(np.argmax(label_vector))  # convert one-hot to int

        img = img.astype(np.uint8)

        img = self.transform(img)

        return img, label



