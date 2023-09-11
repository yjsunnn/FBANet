import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)

import argparse
import options

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss, GWLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from ManualDataset import ManualDatasets, ManualDatasets_validation
from utils.calculate_parameters import calculate_parameters
import torchvision.transforms as transforms

# from utils.loader import  get_training_data,get_validation_data

######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)
print('%s Created, Parameters: %d' % (model_restoration.__class__.__name__, calculate_parameters(model_restoration)))

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - start_epoch + 1, eta_min=1e-6)

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Loss ###########
criterion1 = CharbonnierLoss().cuda()
criterion2 = GWLoss(rgb_range=1.).cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}

######### Load datset ###########
train_dataset = ManualDatasets(root=opt.dataroot, crop_sz=opt.train_ps, burst_size=14, split='train')
val_dataset = ManualDatasets_validation(root=opt.dataroot, crop_sz=opt.train_ps, burst_size=14, split='val')

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.train_workers,
                          drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.eval_workers,
                        drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(train_loader, 0):
        # zero_grad
        print("iteration={}".format(i))
        optimizer.zero_grad()

        target = data['HR'].cuda()
        input_ = data['LR'].cuda()

        # if epoch>5:
        #     target, input_ = utils.MixUp_AUG().aug(target, input_)
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1)
            loss = criterion1(restored, target) + 3*criterion2(restored, target)
        loss_scaler(
            loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss += loss.item()

        #### Evaluation ####
        if (i + 1) % eval_now == 0 and i > 0:
            print("Now in Evaluation Mode!")
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val['HR'].cuda()
                    input_ = data_val['LR'].cuda()
                    filenames = data_val['burst_name']
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)
                    restored = torch.clamp(restored, 0, 1)
                    # transform = transforms.Compose([transforms.ToPILImage()])
                    # for restored_index in range(len(restored)):
                    #     if restored[restored_index].dim() == 3:
                    #         sr_img_saved = transform(restored[restored_index])
                    #         sr_img_saved.save('{}/{}_epoch{}.png'.format(result_dir, filenames[restored_index], epoch))
                    #         print("Image {} saved! restored.max - {}".format(filenames[restored_index],restored[restored_index].max()))
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                print(
                    "[Ep %d it %d\t PSNR Manual: %.4f\t] ----  [best_Ep_Manual %d best_it_Manual %d Best_PSNR_Manual %.4f] " % (
                    epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                with open(logname, 'a') as f:
                    f.write(
                        "[Ep %d it %d\t PSNR Manual: %.4f\t] ----  [best_Ep_Manual %d best_it_Manual %d Best_PSNR_Manual %.4f] " \
                        % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss,
                                                                                scheduler.get_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
print("Now time is : ", datetime.datetime.now().isoformat())
