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


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_list, target,transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_list = img_list
        self.target = target

    def __getitem__(self, index):
        # image = cv2.imread(os.path.join(self.root, self.img_list[index]))#, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # pil_image = PIL.Image.open('Image.jpg').convert('RGB')
        # open_cv_image = numpy.array(pil_image)
        # # Convert RGB to BGR
        # open_cv_image = open_cv_image[:, :, ::-1].copy()

        # When using pytorch, must load image with pillow
        image = Image.open(os.path.join(self.root, self.img_list[index]))
        # Convert RGB to BGR
        image = image.convert('RGB')
        # Convert pillow to cv2
        open_cv_image = np.array(image)
        # plt.imshow(open_cv_image)
        # plt.show()

        image = open_cv_image[:, :, ::-1].copy()
        # plt.imshow(image)
        # plt.show()
        # print(image.shape)


        label = self.target[index]

        # print('self.img_list[index]', self.img_list[index])
        # print('self.target[index]', self.target[index])
        clahe_do = random.randint(0,1)
        if clahe_do == 1:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img1 = clahe.apply(image[:, :, 0])
            img2 = clahe.apply(image[:, :, 1])
            img3 = clahe.apply(image[:, :, 2])

            image = np.stack([img1, img2, img3], axis=2)

        image = cv2.resize(image, (224,224))

        # label = torch.from_numpy(label)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.img_list)

class FaceLandmarksDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

