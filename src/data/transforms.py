import sys
sys.path.append('./')

import torch
import torchvision.transforms.functional as TF
# from data import MedicalDataset
import matplotlib.pyplot as plt
import random


class AdjustBrightness(object):
    """调整图片亮度"""
    def __init__(self, adjust_prob):
        self.adjust_prob = adjust_prob
    
    def __call__(self, image, mask):
        if random.random() < self.adjust_prob:
            brightness_factor = random.uniform(0.5, 1.5)
            image = TF.adjust_brightness(image, brightness_factor=brightness_factor)
        return image, mask
    
class RandomCrop(object):
    """随机裁剪图片"""
    def __init__(self, crop_prob, crop_width, crop_height):
        self.crop_prob = crop_prob
        self.crop_width = crop_width
        self.crop_height = crop_height
    
    def __call__(self, image, mask):
        if random.random() < self.crop_prob:
            width, height = image.size
            # crop_width, crop_height = self.crop_size
            top = random.randint(0, height - self.crop_height)
            left = random.randint(0, width - self.crop_width)

            image = TF.crop(image, top, left, self.crop_height, self.crop_width)
            mask = TF.crop(mask, top, left, self.crop_height, self.crop_width)

        return image, mask

class Flip(object):
    """翻转图像"""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
    
    def __call__(self, image, mask):
        # 水平翻转
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 垂直翻转
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        return image, mask

class Rotate(object):
    def __init__(self, rotate_prob):
        self.rotate_prob = rotate_prob
    
    def __call__(self, image, mask):
        if random.random() < self.rotate_prob:
            random_angle = random.uniform(-180, 180)
            image = TF.rotate(img=image, angle=random_angle)
            mask = TF.rotate(img=mask, angle=random_angle)
        return image, mask

class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, mask):
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

# if __name__ == '__main__':
#     transform = Compose([
#         Flip(0.5),
#         Rotate(0.5),
#         AdjustBrightness(0.5),
#         RandomCrop(0.5, 300, 300),
#         Resize((512, 512)), 
#         ToTensor()
#     ])
    

#     dataset = MedicalDataset(r'D:\Code\UnetMedical\data\数据集\train_val', r'D:\Code\UnetMedical\data\数据集\train_val_mask', transform=transform)
    
#     train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
#     images, masks = next(iter(train_data_loader))
#     print(images.shape, images.dtype)   # torch.Size([8, 1, 512, 512]) torch.float32
#     print(masks.shape, masks.dtype)     # torch.Size([8, 1, 512, 512]) torch.float32
    
#     plt.figure(figsize=(20, 8), dpi=80)
    
#     for idx in range(images.shape[0]):
#         img = images[idx]
#         ax = plt.subplot(2, 8, idx + 1)
#         ax.imshow(images[idx].permute(1, 2, 0))
#         ax.set_title(f'ultrasound_{idx}')

#         ax = plt.subplot(2, 8, idx + 9)
#         ax.imshow(masks[idx].permute(1, 2, 0))
#         ax.set_title(f'mask_{idx}')
    
#     plt.tight_layout()
#     plt.savefig(r'D:\Code\UnetMedical\output\thyroid_ultrasound_by_augment.png')
#     plt.show()
