from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import time
from tqdm import tqdm
from model import MiniUnet
import h5py




def encode_labels(color_mask):
    encode_mask = np.zeros((np.shape(color_mask)[0], np.shape(color_mask)[1]))

    for x in range(np.shape(color_mask)[0]):
        for y in range(np.shape(color_mask)[1]):

            if ((color_mask[x][y][0] == 255) & (color_mask[x][y][1] == 255) & (color_mask[x][y][2] == 0)):
                encode_mask[x][y] = 0.25
            elif ((color_mask[x][y][0] == 0) & (color_mask[x][y][1] == 0) & (color_mask[x][y][2] == 255)):
                encode_mask[x][y] = 0.5

            elif ((color_mask[x][y][0] == 0) & (color_mask[x][y][1] == 255) & (color_mask[x][y][2] == 0)):
                encode_mask[x][y] = 0.75

            elif ((color_mask[x][y][0] == 255) & (color_mask[x][y][1] == 0) & (color_mask[x][y][2] == 0)):
                encode_mask[x][y] = 1.0


    return encode_mask


class CustomMaskDataset(Dataset):
    def __init__(self, path, transform=None):
        self.im = []
        self.mk = []

        self.masks = glob.glob(path + "mask_images_0/*")
        self.files = glob.glob(path + "train_images_0/*")
        print(self.masks)
        # for i in range(len(self.masks)):
        # for i in range(1):
        #    self.files.append(self.masks[i].split('_')[3])
        # self.files = glob.glob(path+"/train_image/*")
        self.files.sort()
        self.masks.sort()
        self.transform = transform
        for i, name in enumerate(self.files):
            image = Image.open(self.files[i])
            image.load()
            self.im.append(image)
            image2 = np.flip(image, 0).copy()
            self.im.append(image2)
            image3 = np.flip(image, 1).copy()
            self.im.append(image3)

        # image.close()
        for i, name in enumerate(self.masks):
            mask = Image.open(self.masks[i])
            mask.load()
            mask = np.array(mask)
            mask = encode_labels(mask)
            self.mk.append(mask)
            image2 = np.flip(mask, 0).copy()
            self.mk.append(image2)
            image3 = np.flip(mask, 1).copy()
            self.mk.append(image3)

    def __getitem__(self, i):
        # PUT YOUR CODE HERE
        if self.transform == None:
            im = self.im[i]
            mask = self.mk[i]
            return im, mask
        else:
            im = self.transform(self.im[i])
            mask = self.transform(self.mk[i])
            return im, mask

    def __len__(self):
        return len(self.im)




def train(model, loss_function, optimizer, epochs=20):
    model.train()

    loss_hist = []
    test_accuracy = []
    train_accuracy = []
    for epoch in range(epochs):
        ep_loss = 0
        for images, labels in tqdm(train_loader):  # get batch
            labels = labels.squeeze(0)
            optimizer.zero_grad()  # sets the gradients of all optimized tensors to zero.
            outputs = model.forward(images)  # call forward inside
            probs = torch.softmax(outputs, dim=1)
            masks_pred = torch.argmax(probs, dim=1)
            loss = loss_function(outputs, labels.type(torch.LongTensor))
            loss.backward()  # calculate gradients
            optimizer.step()  # performs a single optimization step (parameter update).
            ep_loss += loss.item()
        loss_hist.append(ep_loss / len(train_loader))
        print(f"Epoch={epoch} loss={loss_hist[epoch]:.4}")
    #  test_accuracy.append(validate(model, testloader))
    #   train_accuracy.append(validate(model, trainloader))

    return loss_hist


if __name__ == '__main__':
    path = 'C:/Users/filod/PycharmProjects/Sevestal/data/'
    DATA_FILE = 'dataset.h5'
    IMG_SIZE = 240
    IMG_CHANNELS = 3

    # Specify shape
    train_shape = (5, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    y_shape = (5, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    hdf5_file = h5py.File(DATA_FILE, mode='w')
    hdf5_file.create_dataset('x_train', train_shape, np.uint8, compression="gzip")
    hdf5_file.create_dataset('y_train', y_shape, np.uint8, compression="gzip")
    print('check02')
    train_dataset = CustomMaskDataset(path, transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Resize((240, 240))]))
    print('check2')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model = MiniUnet(n_channels=3, n_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    train(model,  criterion, optimizer, epochs=1)
    torch.save(model.state_dict(), 'model.h5')