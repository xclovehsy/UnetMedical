# import os
# import torch
# from pathlib import Path
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader, random_split


# class DataManager:
#     def __init__(self, train_folder, train_size, batch_size, seed):

#         transform = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#         ])
        
#         dataset = MedicalDataset(train_folder, transform)
#         train_size = int(train_size * len(dataset))
#         test_size = len(dataset) - train_size
        
#         train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#         self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
#     def get_train_dataloader(self):
#         return self.train_dataloader
    
#     def get_valid_dataloader(self):
#         return self.test_dataloader
        

# class MedicalDataset(Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_dir = img_dir
#         self.mask_names = [path.name for path in list(Path(img_dir).rglob('*mask.jpg'))]
#         self.img_names = [name.split('_mask.jpg')[0] + '.jpg' for name in self.mask_names]
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_names)

#     def __getitem__(self, idx):
#         img_name = self.img_names[idx]
#         mask_name = self.mask_names[idx]
        
#         img = Image.open(os.path.join(self.img_dir, img_name))
#         mask = Image.open(os.path.join(self.img_dir, mask_name))
        
#         if self.transform:
#             img = self.transform(img)
#             mask = self.transform(mask)
            
#         return img, mask

        
# if __name__ == '__main__':
#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor(),
#     ])
#     dataset = MedicalDataset(r'D:\Code\UnetMedical\data\数据集\train_val', transform=transform)
#     img, mask = dataset[0]
#     print(img.shape, img.dtype)
#     print(mask.shape, mask.dtype)


import os
import torch
from pathlib import Path
from PIL import Image
from data.transforms import *
from torch.utils.data import Dataset, DataLoader, random_split


class DataManager:
    def __init__(self, config):

        train_transform = Compose([
            Flip(0.5),
            Rotate(0.5),
            AdjustBrightness(0.5),
            RandomCrop(0.5, 300, 300),
            Resize((512, 512)), 
            ToTensor()
        ])

        test_transform = Compose([
            Resize((512, 512)), 
            ToTensor()
        ])
        
        train_dataset = MedicalDataset(config.train_img_dir, config.train_mask_dir, train_transform)
        val_dataset = MedicalDataset(config.val_img_dir, config.val_mask_dir, test_transform)
        test_dataset = MedicalDataset(config.test_img_dir, config.test_mask_dir, test_transform)
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
        
    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_valid_dataloader(self):
        return self.val_dataloader
    
    def get_test_dataloader(self):
        return self.test_dataloader
        

class MedicalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = [path.name for path in list(Path(img_dir).rglob('*'))]
        self.mask_names = [name.split('.jpg')[0] + '_mask.jpg' for name in self.img_names]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_names[idx]))
        mask = Image.open(os.path.join(self.mask_dir, self.mask_names[idx]))
        
        if self.transform:
            img, mask = self.transform(img, mask)
            
        return img, mask

    



