from PIL import Image
import os
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self, root_img, transform=None):
        self.root_img = root_img
        self.transform = transform
        self.images = os.listdir(root_img)
        self.length_dataset = len(self.images)
        
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        img = self.images[index % self.length_dataset]
    
        img_path = os.path.join(self.root_img, img)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            A_img = self.transform(img)

        return A_img  