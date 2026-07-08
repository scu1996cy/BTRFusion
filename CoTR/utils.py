import numpy as np
from PIL import Image
import random
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def save_gray_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = image_numpy[0] / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class Transformer_2D(nn.Module):
    def __init__(self):
        super(Transformer_2D, self).__init__()
    # @staticmethod
    def forward(self,src, flow):
        b = flow.shape[0]
        h = flow.shape[2]
        w = flow.shape[3]

        size = (h,w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b,1,1,1).to(src.device)
        new_locs = grid+flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1 , 0]]
        warped = F.grid_sample(src,new_locs,align_corners=True,padding_mode="border")
        # ctx.save_for_backward(src,flow)
        return warped
