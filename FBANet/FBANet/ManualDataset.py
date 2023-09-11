import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os, random, json
import os.path
from utils import is_png_file, load_img, Augment_RGB_torch
from collections import OrderedDict
import sys
import numpy as np
import skimage.io as io
import pdb
import torch

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


def _mod_crop(im, scala):
    w, h = im.size
    return im.crop((0, 0, w - w % scala, h - h % scala))


def get_crop(img, r1, r2, c1, c2):
    im_raw = img[:, r1:r2, c1:c2]
    return im_raw


# manual datasets
class ManualDatasets(torch.utils.data.Dataset):
    """ Real-world burst super-resolution dataset. """

    def __init__(self, root, crop_sz=64, burst_size=14, center_crop=False, random_flip=False, sift_lr=False,
                 split='train'):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            random_flip: Whether to apply random horizontal and vertical flip
            split: Can be 'train' or 'val'
        """
        assert burst_size <= 14, 'burst_sz must be less than or equal to 14'
        # assert crop_sz <= 80, 'crop_sz must be less than or equal to 80'
        assert split in ['train', 'val']
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.burst_size = burst_size
        self.crop_sz = crop_sz
        self.split = split
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.sift_lr = sift_lr

        self.root = root
        # split trainset and testset in one dir
        if self.split == 'val':
            root = root + '/test'
        else:
            root = root + '/train'

        self.hrdir = root + '/' + 'HR'
        self.lrdir = root + '/' + 'LR_aligned'
        print(self.lrdir)

        self.substract_black_level = True
        self.white_balance = False

        self.burst_list = self._get_burst_list()
        self.data_length = len(self.burst_list)
        # self.data_length = 20

    def _get_burst_list(self):
        burst_list = sorted(os.listdir(self.lrdir))
        # print(burst_list)
        return burst_list

    def _get_raw_image(self, burst_id, im_id):
        # Manual_dataset/train/LR/109_28/109_MFSR_Sony_0028_x4_00.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_number2 = int(self.burst_list[burst_id].split('_')[-1])

        path = '{}/{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(self.lrdir, self.burst_list[burst_id], burst_number,
                                                                burst_number2, im_id)

        image = Image.open(path)  # RGB,W, H, C
        image = self.transform(image)
        # print(image.shape)
        # image = cv2.imread(path, cv2.COLOR_BGR2RGB)
        return image

    def _get_gt_image(self, burst_id):
        # 000_MFSR_Sony_0001_x4.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_nmber2 = int(self.burst_list[burst_id].split('_')[-1])
        path = '{}/{}/{}_MFSR_Sony_{:04d}_x4.png'.format(self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)

        image = Image.open(path)  # RGB,W, H, C
        image = self.transform(image)
        return image

    def get_burst(self, burst_id, im_ids):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]
        # pic = self._get_raw_image(burst_id, 0)
        gt = self._get_gt_image(burst_id)

        return frames, gt

    def _sample_images(self):
        burst_size = self.burst_size
        ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
        ids = [0, ] + ids
        return ids

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        frames, gt = self.get_burst(index, im_ids)
        info = self.get_burst_info(index)

        # Extract crop if needed
        if frames[0].shape[-1] != self.crop_sz:
            r1 = random.randint(0, frames[0].shape[-2] - self.crop_sz)
            c1 = random.randint(0, frames[0].shape[-1] - self.crop_sz)
            r2 = r1 + self.crop_sz
            c2 = c1 + self.crop_sz

            scale_factor = gt.shape[-1] // frames[0].shape[-1]

            # print(scale_factor)

            frames = [get_crop(im, r1, r2, c1, c2) for im in frames]
            gt = get_crop(gt, scale_factor * r1, scale_factor * r2, scale_factor * c1, scale_factor * c2)

        apply_trans = transforms_aug[random.getrandbits(3)]
        frames = [getattr(augment, apply_trans)(im) for im in frames]
        gt = getattr(augment, apply_trans)(gt)

        burst = torch.stack(frames, dim=0)
        burst = burst.float()
        frame_gt = gt.float()

        data = {}
        data['LR'] = burst
        data['HR'] = frame_gt
        data['burst_name'] = info['burst_name']

        return data


class ManualDatasets_validation(torch.utils.data.Dataset):
    """ Real-world burst super-resolution dataset. """

    def __init__(self, root, crop_sz=64, burst_size=14, center_crop=False, random_flip=False, sift_lr=False,
                 split='train'):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            random_flip: Whether to apply random horizontal and vertical flip
            split: Can be 'train' or 'val'
        """
        assert burst_size <= 14, 'burst_sz must be less than or equal to 14'
        # assert crop_sz <= 80, 'crop_sz must be less than or equal to 80'
        assert split in ['train', 'val']
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.burst_size = burst_size
        self.crop_sz = crop_sz
        self.split = split
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.sift_lr = sift_lr

        self.root = root
        # split trainset and testset in one dir
        if self.split == 'val':
            root = root + '/test'
        else:
            root = root + '/train'

        self.hrdir = root + '/' + 'HR'
        self.lrdir = root + '/' + 'LR_aligned'
        print(self.lrdir)

        self.substract_black_level = True
        self.white_balance = False

        self.burst_list = self._get_burst_list()
        self.data_length = len(self.burst_list)
        # self.data_length = 20

    def _get_burst_list(self):
        burst_list = sorted(os.listdir(self.lrdir))
        # print(burst_list)
        return burst_list

    def _get_raw_image(self, burst_id, im_id):
        # Manual_dataset/train/LR/109_28/109_MFSR_Sony_0028_x4_00.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_number2 = int(self.burst_list[burst_id].split('_')[-1])

        path = '{}/{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(self.lrdir, self.burst_list[burst_id], burst_number,
                                                                burst_number2, im_id)

        image = Image.open(path)  # RGB,W, H, C
        image = self.transform(image)
        # print(image.shape)
        # image = cv2.imread(path, cv2.COLOR_BGR2RGB)
        return image

    def _get_gt_image(self, burst_id):
        # 000_MFSR_Sony_0001_x4.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_nmber2 = int(self.burst_list[burst_id].split('_')[-1])
        path = '{}/{}/{}_MFSR_Sony_{:04d}_x4.png'.format(self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)

        image = Image.open(path)  # RGB,W, H, C
        image = self.transform(image)
        return image

    def get_burst(self, burst_id, im_ids):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]
        # pic = self._get_raw_image(burst_id, 0)
        gt = self._get_gt_image(burst_id)

        return frames, gt

    def _sample_images(self):
        burst_size = self.burst_size
        ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
        ids = [0, ] + ids
        return ids

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        frames, gt = self.get_burst(index, im_ids)
        info = self.get_burst_info(index)

        # Extract crop if needed
        if frames[0].shape[-1] != self.crop_sz:
            r1 = random.randint(0, frames[0].shape[-2] - self.crop_sz)
            c1 = random.randint(0, frames[0].shape[-1] - self.crop_sz)
            r2 = r1 + self.crop_sz
            c2 = c1 + self.crop_sz

            scale_factor = gt.shape[-1] // frames[0].shape[-1]

            # print(scale_factor)

            frames = [get_crop(im, r1, r2, c1, c2) for im in frames]
            gt = get_crop(gt, scale_factor * r1, scale_factor * r2, scale_factor * c1, scale_factor * c2)

        burst = torch.stack(frames, dim=0)
        burst = burst.float()
        frame_gt = gt.float()

        data = {}
        data['LR'] = burst
        data['HR'] = frame_gt
        data['burst_name'] = info['burst_name']

        return data

class ManualDatasets_test(torch.utils.data.Dataset):
    """ Real-world burst super-resolution dataset. """

    def __init__(self, root, burst_size=14, center_crop=False, random_flip=False, sift_lr=False,
                 split='train'):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            random_flip: Whether to apply random horizontal and vertical flip
            split: Can be 'train' or 'val'
        """
        # assert burst_size <= 14, 'burst_sz must be less than or equal to 14'
        # assert crop_sz <= 80, 'crop_sz must be less than or equal to 80'
        assert split in ['train', 'val']
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.burst_size = burst_size
        self.split = split
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.sift_lr = sift_lr

        self.root = root
        # split trainset and testset in one dir
        if self.split == 'val':
            root = root + '/test'
        else:
            root = root + '/train'

        self.hrdir = root + '/' + 'HR'
        self.lrdir = root + '/' + 'LR_aligned'
        print(self.lrdir)

        self.substract_black_level = True
        self.white_balance = False

        self.burst_list = self._get_burst_list()
        self.data_length = len(self.burst_list)
        # self.data_length = 20

    def _get_burst_list(self):
        burst_list = sorted(os.listdir(self.lrdir))
        # print(burst_list)
        return burst_list

    def _get_raw_image(self, burst_id, im_id):
        burst_number2 = int(self.burst_list[burst_id])

        path = '{}/{}/MFSR_Sony_{:04d}_x1_{:02d}.png'.format(self.lrdir, self.burst_list[burst_id], burst_number2, im_id)

        image = Image.open(path)  # RGB,W, H, C
        image = self.transform(image)
        # print(image.shape)
        # image = cv2.imread(path, cv2.COLOR_BGR2RGB)
        return image

#     def _get_gt_image(self, burst_id):
#         # 000_MFSR_Sony_0001_x4.png
#         burst_number = self.burst_list[burst_id].split('_')[0]
#         burst_nmber2 = int(self.burst_list[burst_id].split('_')[-1])

#         path = '{}/{}/{}_MFSR_Sony_{:04d}_x4.png'.format(self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)

#         image = Image.open(path)  # RGB,W, H, C
#         image = self.transform(image)
#         return image

    def get_burst(self, burst_id, im_ids):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]
        # pic = self._get_raw_image(burst_id, 0)
#         gt = self._get_gt_image(burst_id)

        return frames

    def _sample_images(self):
        burst_size = self.burst_size
        ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
        ids = [0, ] + ids
#         ids = []
#         for i in range(0, 14):
#             ids.append(0)
        return ids

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        frames = self.get_burst(index, im_ids)
        info = self.get_burst_info(index)

        burst = torch.stack(frames, dim=0)
        burst = burst.float()
#         frame_gt = gt.float()

        data = {}
        data['LR'] = burst
#         data['HR'] = frame_gt
        data['burst_name'] = info['burst_name']

        return data