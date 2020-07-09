import os
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from seg_train.data_load_seg_augmentation import myOwnSegDataset
import torchsummary as summary
from seg_train.unet_model import UNet

train_path = "/media/j/DATA/dataset/hemorrhage/down_sampling/train"
val_path = "/media/j/DATA/dataset/hemorrhage/down_sampling/val"

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

##### data load using own dataset

data_transforms = {
    'train':
    transforms.Compose([
        transforms.ToTensor(),
    ]),
    'validation':
    transforms.Compose([
        transforms.ToTensor(),
    ]),
}

my_dataset = myOwnSegDataset(img_path=train_path + '/' + 'img',
                             transforms=data_transforms['train'],mask_path=train_path + '/' + 'mask', img_list=os.listdir(train_path + '/' + 'img'))
val_my_dataset = myOwnSegDataset(img_path=val_path + '/' + 'img',
                                 transforms=data_transforms['validation'],mask_path=val_path + '/' + 'mask', img_list=os.listdir(train_path + '/' + 'img'))

dataloaders = {
    'train':
    torch.utils.data.DataLoader(my_dataset,
                                batch_size=4,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'validation':
    torch.utils.data.DataLoader(val_my_dataset,
                                batch_size=4,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(3,1).to(device)

# summary(model, (3, 256, 256))

for param in model.parameters():
    param.requires_grad = True

print(model)

criterion = nn.BCELoss().cuda()
optimizer = optim.Adam(model.parameters())

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

torch.save(model_trained.state_dict(), '/media/j/DATA/pytorch_temp_ckpt/weights.pth')
