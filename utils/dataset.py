'''Dataset and DataLoader'''
import torch, os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class Dataset(Dataset):
    def __init__(self, path, im_size = 128, lim = 10):
        # im_size = Resized image size
        # lim = first n images in the directory will be used. If all images are used GPU usage is intense.
        self.sizes = [im_size, im_size]
        items, labels = [], []

        for data in os.listdir(path)[:lim]:
            # path = './img_align_celeba'
            # data = 023453.jpg (for example)
            item = os.path.join(path,data)
            items.append(item)
            labels.append(data)

        self.items = items
        self.labels = labels
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        data = Image.open(self.items[index]).convert('RGB')    # PIL Image
        data = np.asarray(transforms.Resize(self.sizes)(data)) # [128,128,3]
        data = torch.from_numpy(data).div(255).permute(2,0,1) # values between [0 and 1] with shape [3,128,128]
        return data, self.labels[index]
