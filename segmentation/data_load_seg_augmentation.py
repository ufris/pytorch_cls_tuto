import os, random
import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2
import numpy as np
from skimage import io, transform
from PIL import Image

class myOwnSegDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, mask_path, img_list ,transforms=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transforms = transforms
        self.img_list = img_list

    def __getitem__(self, index):
        # When using pytorch, must load image with pillow
        image = Image.open(os.path.join(self.img_path, self.img_list[index]))
        # Convert RGB to BGR
        image = image.convert('RGB')
        # Convert pillow to cv2
        open_cv_image = np.array(image)

        image = open_cv_image[:, :, ::-1].copy()

        label = Image.open(os.path.join(self.mask_path, self.img_list[index])).convert('L')
        label = np.array(label)

        print(os.path.join(self.img_path, self.img_list[index]))
        print(os.path.join(self.mask_path, self.img_list[index]))

        # print('self.img_list[index]', self.img_list[index])
        # print('self.target[index]', self.target[index])
        clahe_do = random.randint(0,1)
        if clahe_do == 1:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img1 = clahe.apply(image[:, :, 0])
            img2 = clahe.apply(image[:, :, 1])
            img3 = clahe.apply(image[:, :, 2])

            image = np.stack([img1, img2, img3], axis=2)

        image = cv2.resize(image, (256,256))
        label = cv2.resize(label, (256, 256))
        _, label = cv2.threshold(label,0,255,cv2.THRESH_BINARY)

        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)

        return image / 255.0 , label

    def __len__(self):
        return len(self.img_list)
