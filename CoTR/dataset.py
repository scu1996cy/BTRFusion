from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        a_transform_list = [transforms.ToTensor()]
        a_transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        b_transform_list = [transforms.Grayscale(1)]
        b_transform_list += [transforms.ToTensor()]
        b_transform_list += [transforms.Normalize((0.5,), (0.5,))]

        self.a_transform = transforms.Compose(a_transform_list)
        self.b_transform = transforms.Compose(b_transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')

        a = self.a_transform(a)
        b = self.b_transform(b)

        return a, b

    def __len__(self):
        return len(self.image_filenames)