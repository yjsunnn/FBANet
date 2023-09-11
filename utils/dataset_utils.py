import torch
import os
from torch.nn.functional import pad as tensor_pad


def tensor_divide_burst(tensor_burst, psize, overlap, pad=True):
    """
    Divide Tensor Into Blocks, Especially for Remainder
    :param tensor:
    :param psize:
    :param overlap:
    :return: List
    """
    B, T, C, H, W = tensor_burst.shape

    # Pad to number that can be divisible
    if pad:
        h_pad = psize - H % psize if H % psize != 0 else 0
        w_pad = psize - W % psize if W % psize != 0 else 0
        H += h_pad
        W += w_pad
        if h_pad != 0 or w_pad != 0:
            padded_tensor = []
            for frames in range(0, T):
                current_frame = tensor_burst[:, frames, :, :, :]
                padded_frame = tensor_pad(current_frame, (0, w_pad, 0, h_pad), mode='reflect').data
                padded_tensor.append(padded_frame)  # List of T*[B,C,H,W]; Finally we want [B,T,C,H,W]

            tensor = torch.zeros(B, T, C, H, W)
            for frames in range(0, T):
                tensor[:, frames, :, :, :] = padded_tensor[frames]

            print("tensor.shape:", tensor.shape)
            # tensor = tensor_pad(tensor_burst, (0, w_pad, 0, h_pad), mode='reflect').data

    h_block = H // psize
    w_block = W // psize
    blocks = []

    tensor_copy = tensor
    tensor = torch.zeros(B, T, C, H + 2 * overlap, W + 2 * overlap)

    if overlap != 0:
        for batch in range(0, B):
            tensor[batch, :, :, :, :] = tensor_pad(tensor_copy[batch, :, :, :, :], (overlap, overlap, overlap, overlap),
                                                   mode='reflect').data  # tensor.shape=[T,C,H+2*overlap,W+2*overlap]
    else:
        print("Error! Check overlap and tensor_burst! Tensor burst should be [B,T,C,H,W] in testing!")

    for i in range(h_block):
        for j in range(w_block):
            end_h = tensor.shape[3] if i + 1 == h_block else (i + 1) * psize + 2 * overlap
            end_w = tensor.shape[4] if j + 1 == w_block else (j + 1) * psize + 2 * overlap
            # end_h = (i + 1) * psize + 2 * overlap
            # end_w = (j + 1) * psize + 2 * overlap
            part = tensor[:, :, :, i * psize: end_h, j * psize: end_w]
            blocks.append(part)
    return blocks


def tensor_merge_burst(blocks, tensor, psize, overlap, pad=True):
    """
    Combine many small patch into one big Image
    :param blocks: List of 4D Tensors or just a 4D Tensor
    :param tensor:  has the same size as the big image
    :param psize:
    :param overlap:
    :return: Tensor
    """
    B, C, H, W = tensor.shape

    # Pad to number that can be divisible
    if pad:
        h_pad = psize - H % psize if H % psize != 0 else 0
        w_pad = psize - W % psize if W % psize != 0 else 0
        H += h_pad
        W += w_pad

    tensor_new = torch.FloatTensor(B, C, H, W)
    h_block = H // psize
    w_block = W // psize
    # print(tensor.shape, tensor_new.shape)
    for i in range(h_block):
        for j in range(w_block):
            end_h = tensor_new.shape[2] if i + 1 == h_block else (i + 1) * psize
            end_w = tensor_new.shape[3] if j + 1 == w_block else (j + 1) * psize
            # end_h = (i + 1) * psize
            # end_w = (j + 1) * psize
            part = blocks[i * w_block + j]

            if len(part.shape) < 4:
                part = part.unsqueeze(0)

            tensor_new[:, :, i * psize: end_h, j * psize: end_w] = \
                part[:, :, overlap: part.shape[2] - overlap, overlap: part.shape[3] - overlap]

    # Remove Pad Edges
    B, C, H, W = tensor.shape
    tensor_new = tensor_new[:, :, :H, :W]
    return tensor_new


def tensor_divide(tensor, psize, overlap, pad=True):
    """
    Divide Tensor Into Blocks, Especially for Remainder
    :param tensor:
    :param psize:
    :param overlap:
    :return: List
    """
    B, C, H, W = tensor.shape

    # Pad to number that can be divisible
    if pad:
        h_pad = psize - H % psize if H % psize != 0 else 0
        w_pad = psize - W % psize if W % psize != 0 else 0
        H += h_pad
        W += w_pad
        if h_pad != 0 or w_pad != 0:
            tensor = tensor_pad(tensor, (0, w_pad, 0, h_pad), mode='reflect').data

    h_block = H // psize
    w_block = W // psize
    blocks = []
    if overlap != 0:
        tensor = tensor_pad(tensor, (overlap, overlap, overlap, overlap), mode='reflect').data

    for i in range(h_block):
        for j in range(w_block):
            end_h = tensor.shape[2] if i + 1 == h_block else (i + 1) * psize + 2 * overlap
            end_w = tensor.shape[3] if j + 1 == w_block else (j + 1) * psize + 2 * overlap
            # end_h = (i + 1) * psize + 2 * overlap
            # end_w = (j + 1) * psize + 2 * overlap
            part = tensor[:, :, i * psize: end_h, j * psize: end_w]
            blocks.append(part)
    return blocks


def tensor_merge(blocks, tensor, psize, overlap, pad=True):
    """
    Combine many small patch into one big Image
    :param blocks: List of 4D Tensors or just a 4D Tensor
    :param tensor:  has the same size as the big image
    :param psize:
    :param overlap:
    :return: Tensor
    """
    B, C, H, W = tensor.shape

    # Pad to number that can be divisible
    if pad:
        h_pad = psize - H % psize if H % psize != 0 else 0
        w_pad = psize - W % psize if W % psize != 0 else 0
        H += h_pad
        W += w_pad

    tensor_new = torch.FloatTensor(B, C, H, W)
    h_block = H // psize
    w_block = W // psize
    # print(tensor.shape, tensor_new.shape)
    for i in range(h_block):
        for j in range(w_block):
            end_h = tensor_new.shape[2] if i + 1 == h_block else (i + 1) * psize
            end_w = tensor_new.shape[3] if j + 1 == w_block else (j + 1) * psize
            # end_h = (i + 1) * psize
            # end_w = (j + 1) * psize
            part = blocks[i * w_block + j]

            if len(part.shape) < 4:
                part = part.unsqueeze(0)

            tensor_new[:, :, i * psize: end_h, j * psize: end_w] = \
                part[:, :, overlap: part.shape[2] - overlap, overlap: part.shape[3] - overlap]

    # Remove Pad Edges
    B, C, H, W = tensor.shape
    tensor_new = tensor_new[:, :, :H, :W]
    return tensor_new


### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass

    def transform0(self, torch_tensor):
        return torch_tensor

    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1, -2])
        return torch_tensor

    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1, -2])
        return torch_tensor

    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1, -2])
        return torch_tensor

    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor

    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1, -2])).flip(-2)
        return torch_tensor


### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1 - lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1 - lam) * rgb_noisy2

        return rgb_gt, rgb_noisy
