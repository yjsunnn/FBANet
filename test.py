import numpy as np
import os, sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

import scipy.io as sio
from utils.loader import get_validation_data
import utils

from model import BaseModel

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
# from skimage.metrics import structural_similarity as ssim_loss
from ManualDataset import ManualDatasets, ManualDatasets_validation
from utils.image_utils import myPSNR_version2, mySSIM, myLPIPS
from torchvision.transforms import transforms

parser = argparse.ArgumentParser(description='RGB super-resolution test')
parser.add_argument('--input_dir', default='/userhome/aimia/sunyj/Dataset/DRealBSR_RGB',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Motion_MFSR_same_0.0/',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='/userhome/aimia/sunyj/FBANet_20230910/pretrained/model_best.pth',
                    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0,1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='BaseModel', type=str, help='arch')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=64, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=160, help='patch size of training sample')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = ManualDatasets_validation(root=args.input_dir, crop_sz=args.train_ps, burst_size=14, split='val')
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()
with torch.no_grad():
    lpips_val = []
    psnr_val = []
    ssim_val = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test['HR'].cuda()
        rgb_noisy = data_test['LR'].cuda()
        filenames = data_test['burst_name']

        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored, 0, 1)

        psnr_val.append(utils.batch_PSNR(rgb_restored, rgb_gt, False).item())

        lpips_val.append(myLPIPS(rgb_restored, rgb_gt))
        ssim_val.append(mySSIM(rgb_restored, rgb_gt))

        if args.save_images:
            transform = transforms.Compose([transforms.ToPILImage()])
            for restored_index in range(len(rgb_restored)):
                if rgb_restored[restored_index].dim() == 3:
                    sr_img_saved = transform(rgb_restored[restored_index])
                    sr_img_saved.save('{}/{}_same_0.0.png'.format(args.result_dir, filenames[restored_index]))

lpips_val = sum(lpips_val) / len(test_loader)
ssim_val = sum(ssim_val) / len(test_loader)
psnr_val = sum(psnr_val) / len(test_dataset)
# print("PSNR: %f, PSNR_v2: %f, SSIM: %f " %(psnr_val_rgb,psnr_val,ssim_val_rgb))
print("PSNR_v2: %f SSIM: %f LPIPS: %f" % (psnr_val, ssim_val, lpips_val))

