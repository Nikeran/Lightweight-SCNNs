import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as _transforms

class CIFAR10C(Dataset):
    """
    CIFAR-10-C corruptions dataset loader.
    Expects .npy files: data_dir/<corruption>.npy and labels.npy
    """
    def __init__(self, data_dir, corruption='gaussian_noise', severity=1, transform=None):
        arr = np.load(os.path.join(data_dir, f"{corruption}.npy"))
        self.data = arr[(severity-1)*10000:severity*10000]
        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = _transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label