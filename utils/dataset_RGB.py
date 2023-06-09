import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageOps
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'rain')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'norain'))) 

        self.inp_filenames = [os.path.join(rgb_dir, 'rain', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'norain', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = cv2.imread(inp_path)
        inp_img = TF.to_tensor(inp_img)  # 转换成tensor格式
        tar_img = cv2.imread(tar_path)
        tar_img = TF.to_tensor(tar_img)  # 转换成tensor格式
        # inp_img = ImageOps.exif_transpose(inp_img)  # 恢复正常角度的图像
        # tar_img = ImageOps.exif_transpose(tar_img)  # 恢复正常角度的图像


        _,w,h = tar_img.shape
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padh,padw), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padh,padw), padding_mode='reflect')

        # inp_img = TF.to_tensor(inp_img)
        # tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps-1)
        cc     = random.randint(0, ww-ps-1)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)

        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'rain')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'norain')))

        self.inp_filenames = [os.path.join(rgb_dir, 'rain', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'norain', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = cv2.imread(inp_path)
        inp_img = TF.to_tensor(inp_img)  # 转换成tensor格式
        tar_img = cv2.imread(tar_path)
        tar_img = TF.to_tensor(tar_img)  # 转换成tensor格式
        # inp_img = Image.open(inp_path)
        # inp_img = ImageOps.exif_transpose(inp_img)  # 恢复正常角度的图像
        # tar_img = Image.open(tar_path)
        # tar_img = ImageOps.exif_transpose(tar_img)  # 恢复正常角度的图像

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        # inp_img = TF.to_tensor(inp_img)
        # tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

# class DataLoaderTest(Dataset):
#     def __init__(self, inp_dir, img_options):
#         super(DataLoaderTest, self).__init__()
#
#         inp_files = sorted(os.listdir(inp_dir))
#         self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]
#
#         self.inp_size = len(self.inp_filenames)
#         self.img_options = img_options
#
#     def __len__(self):
#         return self.inp_size
#
#     def __getitem__(self, index):
#
#         path_inp = self.inp_filenames[index]
#         filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
#         inp = Image.open(path_inp)
#
#         inp = TF.to_tensor(inp)
#         return inp, filename


def DataLoader_Test(test_dir):
    inp_files = sorted(os.listdir(os.path.join(test_dir, 'rain')))
    tar_files = sorted(os.listdir(os.path.join(test_dir, 'norain')))
    inp_filenames = [os.path.join(test_dir, 'rain', x) for x in inp_files if is_image_file(x)]
    tar_filenames = [os.path.join(test_dir, 'norain', x) for x in tar_files if is_image_file(x)]
    return inp_filenames, tar_filenames

def DataLoader_TestReal(test_dir):
    inp_files = sorted(os.listdir(os.path.join(test_dir, 'input')))
    inp_filenames = [os.path.join(test_dir, 'input', x) for x in inp_files if is_image_file(x)]
    return inp_filenames


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'rain')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'norain')))

        self.inp_filenames = [os.path.join(rgb_dir, 'rain', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'norain', x) for x in tar_files if is_image_file(x)]

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        inp_img = ImageOps.exif_transpose(inp_img)  # 恢复正常角度的图像
        tar_img = Image.open(tar_path)
        tar_img = ImageOps.exif_transpose(tar_img)  # 恢复正常角度的图像
        # print("inp_path: ", inp_path)
        # print("tar_path: ", tar_path)


        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

