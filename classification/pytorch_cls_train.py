import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, random, cv2
import torch
from torchvision import datasets, models, transforms
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from data_load_augmentation import myOwnDataset
import torchsummary as summary

# input_path = "/media/j/DATA/dataset/alien_vs_predator_thumbnails/data/"

dog_path = '/media/j/DATA/dataset/dog_cat/all_dog_cat/dog' + '/'
cat_path = '/media/j/DATA/dataset/dog_cat/all_dog_cat/cat' + '/'

dog_list = ['dog/' + i for i in os.listdir(dog_path)]
cat_list = ['cat/' + i for i in os.listdir(cat_path)]
img_list = dog_list + cat_list

total_cnt = len(img_list)
train_cnt = int(total_cnt * 0.8)

dog_target = [0 for i in range(len(dog_list))]
cat_target = [1 for i in range(len(cat_list))]
target_list = dog_target + cat_target

random.seed(1)
random.shuffle(img_list)
random.seed(1)
random.shuffle(target_list)

print(img_list[:10])
print(target_list[:10])

train_img_list = img_list[:train_cnt]
val_img_list = img_list[train_cnt:]

train_target_list = target_list[:train_cnt]
val_target_list = target_list[train_cnt:]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

##### data load using ImageFolder

# data_transforms = {
#     'train':
#     transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#     ]),
#     'validation':
#     transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         normalize
#     ]),
# }
# image_datasets = {
#     'train':
#     datasets.ImageFolder(input_path + 'train', data_transforms['train']),
#     'validation':
#     datasets.ImageFolder(input_path + 'validation', data_transforms['validation'])
# }
#
# dataloaders = {
#     'train':
#     torch.utils.data.DataLoader(image_datasets['train'],
#                                 batch_size=32,
#                                 shuffle=True,
#                                 num_workers=0),  # for Kaggle
#     'validation':
#     torch.utils.data.DataLoader(image_datasets['validation'],
#                                 batch_size=32,
#                                 shuffle=False,
#                                 num_workers=0)  # for Kaggle
# }

##### data load using own dataset

data_transforms = {
    'train':
    transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]),
}

my_dataset = myOwnDataset(root='/media/j/DATA/dataset/dog_cat/all_dog_cat',transforms=data_transforms['train'],img_list=train_img_list[:1000],target=train_target_list[:1000])
val_my_dataset = myOwnDataset(root='/media/j/DATA/dataset/dog_cat/all_dog_cat',transforms=data_transforms['validation'],img_list=val_img_list[:1000],target=val_target_list[:1000])

dataloaders = {
    'train':
    torch.utils.data.DataLoader(my_dataset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'validation':
    torch.utils.data.DataLoader(val_my_dataset,
                                batch_size=32,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}

#### show img

# def imshow(img):
#     # img = img / 2 + 0.5     # unnormalize
#     # npimg = img.numpy()
#     # plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#     print('#######3',img.shape)
#     img = img.numpy()
#     img = np.transpose(img, (1, 2, 0))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.show()

# 학습용 이미지를 무작위로 가져오기
# dataiter = iter(data_loader)
# images = dataiter.next()

# 이미지 보여주기
# imshow(images[0])
# 정답(label) 출력
# print(' '.join('%5s' % classes[labels[j]] for j in range(1)))


##### It don't use yet
# inputs, masks = next(iter(dataloaders['train']))
# print(inputs.shape, masks.shape)
# plt.imshow(reverse_transform(inputs[3]))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True).to(device)
box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

summary(box_model, input_size=(3, 224, 224))

for param in model.parameters():
    param.requires_grad = True

print(model)

model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2),
    ).to(device)

print(model)

prob = nn.Softmax()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.fc.parameters())

# cnt_dict = {'train':len(train_img_list[:1000]), 'validation':len(val_img_list[:1000])}

def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train','validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # print(inputs.shape)
                # print(labels)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / 1000 #len(my_dataset)

            epoch_acc = running_corrects.double() / 1000 # len(val_my_dataset)

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return model

model_trained = train_model(model, criterion, optimizer, num_epochs=10)

torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5')
