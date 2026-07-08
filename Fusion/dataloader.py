import os
import cv2
import torch.utils.data as data
from args_setting import args

class TrainData(data.Dataset):
    def __init__(self, transform=None):
        super(TrainData, self).__init__()
        self.dir_prefix = './datasets/' + args.patch + '/train/'
        self.img1_dir = os.listdir(self.dir_prefix + args.task.split('-')[0] + '/')
        self.img2_dir = os.listdir(self.dir_prefix + 'MRI/')

        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        assert len(self.img1_dir) == len(self.img2_dir)
        return len(self.img1_dir)

    def __getitem__(self, index):
        self.img1_dir.sort()
        self.img2_dir.sort()
        if args.task == 'CT-MRI':
            img_type1 = args.task.split('-')[0] + '/'
            img1 = cv2.imread(self.dir_prefix + img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
        else:
            img_type1 = args.task.split('-')[0] + '/'
            img1 = cv2.imread(self.dir_prefix + img_type1 + self.img1_dir[index])
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
            img1 = img1[:, :, 0:1]
            img1 = img1.squeeze()
        img2 = cv2.imread(self.dir_prefix + 'MRI/' + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2  # 1,256,256

class TestData(data.Dataset):
    def __init__(self, transform=None):
        super(TestData, self).__init__()
        self.transform = transform
        self.dir_prefix = './datasets/ori/test/'

        self.img1_dir = os.listdir(self.dir_prefix + 'PET/')
        self.img2_dir = os.listdir(self.dir_prefix + 'MRI/')

    def __getitem__(self, index):
        self.img1_dir.sort()
        self.img2_dir.sort()
        img_name = str(self.img1_dir[index])
        img1 = cv2.imread(self.dir_prefix + 'PET/' + self.img1_dir[index])
        img2 = cv2.imread(self.dir_prefix + 'MRI/' + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)

        img1_Y = img1[:, :, 0:1]
        img1_CrCb = img1[:, :, 1:3].transpose(2, 0, 1)

        if self.transform:
            img1_Y = self.transform(img1_Y)
            img2 = self.transform(img2)

        return img_name, img1_Y, img2, img1_CrCb

    def __len__(self):
        assert len(self.img1_dir) == len(self.img2_dir)
        return len(self.img1_dir)
