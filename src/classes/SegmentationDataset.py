
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None, image_ext=".png"):
        
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_ext = image_ext
        
        self.image_list = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(image_ext) and not f.endswith("_mask" + image_ext)
        ])
        
        # self.cache = []
        # for img_name in self.image_list:
        #     # toTensor = transforms.ToTensor()
        #     img = Image.open(os.path.join(root_dir, img_name)).convert("RGB")
            
        #     mask_name = img_name.replace(self.image_ext, "_mask" + self.image_ext)
        #     mask = Image.open(os.path.join(root_dir, mask_name)).convert("L")
        #     # Convert to NumPy array
        #     mask_np = np.array(mask)
        #     # Normalize to 0 / 255
        #     mask_np = np.where(mask_np > 128, 255, 0).astype(np.uint8)
        #     # Convert back to PIL Image
        #     mask_normalized = Image.fromarray(mask_np)
        #     self.cache.append((img, mask_normalized))
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        
        # image, mask = self.cache[idx]
        image_name = self.image_list[idx]
        image = Image.open(os.path.join(self.root_dir, image_name)).convert("RGB")
        
        mask_name = image_name.replace(self.image_ext, "_mask" + self.image_ext)
        mask = Image.open(os.path.join(self.root_dir, mask_name)).convert("L")
        
        # Convert to NumPy array
        mask = np.array(mask)
        # Normalize to 0 / 255
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        # Convert back to PIL Image
        mask = Image.fromarray(mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask