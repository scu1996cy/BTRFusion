import os
import numpy as np
from PIL import Image
import random

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
def crop_and_synchronized_augment(input_folder_A, input_folder_B, output_folder_C, output_folder_D, patch_size=128,
                                  stride=16):
    """
    将文件夹 A 和 B 的对应图片裁剪为固定大小的 patch，随机应用相同的增强，并分别保存到文件夹 C 和 D。

    :param input_folder_A: 输入文件夹 A 的路径
    :param input_folder_B: 输入文件夹 B 的路径
    :param output_folder_C: 输出文件夹 C 的路径
    :param output_folder_D: 输出文件夹 D 的路径
    :param patch_size: 每个 patch 的大小（默认128）
    :param stride: 裁剪步幅（默认16）
    """
    if not os.path.exists(output_folder_C):
        os.makedirs(output_folder_C)
    if not os.path.exists(output_folder_D):
        os.makedirs(output_folder_D)

    for file_name in os.listdir(input_folder_A):
        if file_name.endswith('.png'):
            file_path_A = os.path.join(input_folder_A, file_name)
            file_path_B = os.path.join(input_folder_B, file_name)

            # 确保对应文件存在
            if not os.path.exists(file_path_B):
                print(f"Corresponding file for {file_name} not found in {input_folder_B}. Skipping.")
                continue

            # 打开两张对应的图片
            img_A = Image.open(file_path_A)
            img_B = Image.open(file_path_B)
            img_array_A = np.array(img_A)
            img_array_B = np.array(img_B)

            h, w = img_array_A.shape[:2]
            patch_num = 0

            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    # 裁剪出一个 patch
                    patch_A = img_array_A[i:i + patch_size, j:j + patch_size]
                    patch_B = img_array_B[i:i + patch_size, j:j + patch_size]

                    # 为当前 patch 随机选择一种增强模式
                    #mode = random.randint(0, 7)
                    #augmented_patch_A = augment_img(patch_A, mode=mode)
                    #augmented_patch_B = augment_img(patch_B, mode=mode)

                    # 保存增强后的 patch
                    patch_num += 1
                    patch_name_A = f"{os.path.splitext(file_name)[0]}_{patch_num}.png"
                    patch_name_B = f"{os.path.splitext(file_name)[0]}_{patch_num}.png"
                    patch_path_A = os.path.join(output_folder_C, patch_name_A)
                    patch_path_B = os.path.join(output_folder_D, patch_name_B)

                    #Image.fromarray(augmented_patch_A).save(patch_path_A)
                    #Image.fromarray(augmented_patch_B).save(patch_path_B)
                    Image.fromarray(patch_A).save(patch_path_A)
                    Image.fromarray(patch_B).save(patch_path_B)

    print(f"Processed {input_folder_A} and {input_folder_B} -> {output_folder_C} and {output_folder_D}")

folder_A = './datasets/ori/train/MRI'
folder_B = './datasets/ori/train/PET'

folder_C = './datasets/crop/train/MRI'
folder_D = './datasets/crop/train/PET'

# 执行裁剪并保存
crop_and_synchronized_augment(folder_A, folder_B, folder_C, folder_D, patch_size=128, stride=16)
