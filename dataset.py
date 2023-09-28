from torch.utils.data import Dataset
import os
import csv
import PIL
import torch
import torchvision



class ImageNetValDataset(Dataset):
    def __init__(self, transform, transform_crop_resize, image_folder_path=""):
        self.transform = transform
        self.transform_crop_resize = transform_crop_resize
        self.ToTensor = torchvision.transforms.ToTensor()
        self.image_folder_path = image_folder_path
        self.files = os.listdir(image_folder_path)


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        original_img = PIL.Image.open(self.image_folder_path+self.files[index])

        if self.transform != None:
            img = self.ToTensor(original_img)
            if img.shape[0] == 1:
                img = torch.stack([img[0], img[0], img[0]])
            img = self.transform(img)

        original_img = self.ToTensor(original_img)
        if original_img.shape[0] == 1:
            original_img = torch.stack([original_img[0], original_img[0], original_img[0]])
        original_img = self.transform_crop_resize(original_img)

        return img, original_img



class ImageNetValDataset_500(Dataset):
    def __init__(self, transform, transform_crop_resize, image_folder_path=""):
        self.transform = transform
        self.transform_crop_resize = transform_crop_resize
        self.ToTensor = torchvision.transforms.ToTensor()
        self.image_folder_path = image_folder_path
        self.files = os.listdir(image_folder_path)


    def __len__(self):
        return int(len(self.files)/100)


    def __getitem__(self, index):
        original_img = PIL.Image.open(self.image_folder_path+self.files[index*100])

        if self.transform != None:
            img = self.ToTensor(original_img)
            if img.shape[0] == 1:
                img = torch.stack([img[0], img[0], img[0]])
            img = self.transform(img)

        original_img = self.ToTensor(original_img)
        if original_img.shape[0] == 1:
            original_img = torch.stack([original_img[0], original_img[0], original_img[0]])
        original_img = self.transform_crop_resize(original_img)

        return img, original_img



class AFHQ_Dataset(Dataset):
    def __init__(self, transform=None, image_folder_path=""):
        self.transform = transform
        self.ToTensor = torchvision.transforms.ToTensor()

        self.image_folder_path = image_folder_path
        self.files = os.listdir(self.image_folder_path)


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        img = PIL.Image.open(self.image_folder_path + self.files[index])
        normal_img = self.ToTensor(img)

        if self.transform != None:
            img = self.transform(img)

        return img, normal_img












