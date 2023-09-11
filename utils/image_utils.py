import torch
import numpy as np
import pickle
import cv2
import torch.nn as nn
import torch.nn.functional as F
import math
import lpips
from .mssim import msssim as msssim


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict


def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)


def load_npy(filepath):
    img = np.load(filepath)
    return img


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.
    return img


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps


class PixelWiseError(nn.Module):
    """ Computes pixel-wise error using the specified metric. Optionally boundary pixels are ignored during error
        calculation """

    def __init__(self, metric='l1', boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

        if metric == 'l1':
            self.loss_fn = F.l1_loss
        elif metric == 'l2':
            self.loss_fn = F.mse_loss
        elif metric == 'l2_sqrt':
            def l2_sqrt(pred, gt):
                return (((pred - gt) ** 2).sum(dim=-3)).sqrt().mean()

            self.loss_fn = l2_sqrt
        elif metric == 'charbonnier':
            def charbonnier(pred, gt):
                eps = 1e-3
                return ((pred - gt) ** 2 + eps ** 2).sqrt().mean()

            self.loss_fn = charbonnier
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore,
                        self.boundary_ignore:-self.boundary_ignore]

        # Valid indicates image regions which should be used for loss calculation
        if valid is None:
            err = self.loss_fn(pred, gt)
        else:
            err = self.loss_fn(pred, gt, reduction='none')

            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        return err


class PSNR_class(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = PixelWiseError(metric='l2', boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        pred = pred.cpu()
        gt = gt.cpu()

        mse = self.l2(pred, gt, valid=valid)

        if getattr(self, 'max_value', 1.0) is not None:
            psnr = 20 * math.log10(getattr(self, 'max_value', 1.0)) - 10.0 * mse.log10()
        else:
            psnr = 20 * gt.max().log10() - 10.0 * mse.log10()

        if torch.isinf(psnr) or torch.isnan(psnr):
            # print('invalid psnr')
            pass

        return psnr

    def forward(self, pred, gt, valid=None):
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]

        psnr_all = [p for p in psnr_all if not (torch.isinf(p) or torch.isnan(p))]

        if len(psnr_all) == 0:
            psnr = 0
        else:
            psnr = sum(psnr_all) / len(psnr_all)
        return psnr


myPSNR_version2 = PSNR_class(boundary_ignore=40)


def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR_version2(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR) / len(PSNR) if average else sum(PSNR)


class SSIM(nn.Module):
    def __init__(self, boundary_ignore=None, use_for_loss=False):
        super().__init__()
        self.ssim = msssim.SSIM(spatial_out=True)
        self.boundary_ignore = boundary_ignore
        self.use_for_loss = use_for_loss

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore,
                        self.boundary_ignore:-self.boundary_ignore]

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        loss = self.ssim(pred, gt)

        if valid is not None:
            valid = valid[..., 5:-5, 5:-5]  # assume window size 11

            eps = 1e-12
            elem_ratio = loss.numel() / valid.numel()
            loss = (loss * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)
        else:
            loss = loss.mean()

        if self.use_for_loss:
            loss = 1.0 - loss
        return loss


class LPIPS(nn.Module):
    def __init__(self, boundary_ignore=None, type='alex', bgr2rgb=False):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.bgr2rgb = bgr2rgb

        if type == 'alex':
            self.loss = lpips.LPIPS(net='alex').cuda()
        elif type == 'vgg':
            self.loss = lpips.LPIPS(net='vgg').cuda()
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.bgr2rgb:
            pred = pred[..., [2, 1, 0], :, :].contiguous()
            gt = gt[..., [2, 1, 0], :, :].contiguous()

        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        loss = self.loss(pred, gt)

        return loss.mean()


mySSIM = SSIM(boundary_ignore=40)
myLPIPS = LPIPS(boundary_ignore=40)