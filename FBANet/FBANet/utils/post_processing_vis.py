import pickle
import cv2
import torch
import numpy as np


def generate_processed_image_channel4(im, meta_data, return_np=False, black_level_substracted=True, external_norm_factor=None,
                                      gamma=True, smoothstep=True, no_white_balance=False):
    im = im * meta_data.get('norm_factor', 16383.0)

    if not meta_data.get('black_level_subtracted', False) and not black_level_substracted:
        im = (im - torch.tensor(meta_data['black_level']).view(4, 1, 1))

    if not meta_data.get('while_balance_applied', False) and not no_white_balance:
        im = im * torch.tensor(meta_data['cam_wb']).view(4, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]

    im_out = im


    if external_norm_factor is None:
        im_out = im_out / (im_out.mean() * 5.0)
    else:
        im_out = im_out / external_norm_factor

    im_out = im_out.clamp(0.0, 1.0)

    if gamma:
        im_out = im_out ** (1.0 / 2.2)

    if smoothstep:
        # Smooth curve
        im_out = 3 * im_out ** 2 - 2 * im_out ** 3

    if return_np:
        im_out = torch.stack((im_out[0, :, :], im_out[1:3, :, :].mean(dim=0), im_out[3, :, :]), dim=0)
        im_out = im_out.permute(1, 2, 0).numpy() * 255.0
        im_out = im_out.astype(np.uint8)
    return im_out


def generate_processed_image_channel3(im, meta_data, return_np=False, black_level_substracted=True, external_norm_factor=None,
                                      gamma=True, smoothstep=True, no_white_balance=False):
    im = im * meta_data.get('norm_factor', 16383.0)

    if not meta_data.get('black_level_subtracted', False) and not black_level_substracted:
        meta_data['black_level'] = torch.from_numpy(np.array(meta_data['black_level']))

        meta_data['black_level'] = torch.stack((meta_data['black_level'][0].float(), meta_data['black_level'][1:3].float().mean(), meta_data['black_level'][3].float()), dim=0)
        im = im - torch.tensor(meta_data['black_level']).view(3, 1, 1)

    if not meta_data.get('while_balance_applied', False) and not no_white_balance:
        meta_data['cam_wb'] = torch.from_numpy(np.array(meta_data['cam_wb']))
        meta_data['cam_wb'] = torch.stack((meta_data['cam_wb'][0].float(), meta_data['cam_wb'][1:3].float().mean(), meta_data['cam_wb'][3].float()), dim=0)
        im = im * torch.tensor(meta_data['cam_wb']).view(3, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]

    im_out = im


    if external_norm_factor is None:
        im_out = im_out / (im_out.mean() * 5.0)
    else:
        im_out = im_out / external_norm_factor
        
    im_out = im_out.clamp(0.0, 1.0)

    if gamma:
        im_out = im_out ** (1.0 / 2.2)

    if smoothstep:
        # Smooth curve
        im_out = 3 * im_out ** 2 - 2 * im_out ** 3

    if return_np:
        im_out = im_out.permute(1, 2, 0).numpy() * 255.0
        im_out = im_out.astype(np.uint8)

    return im_out


# if __name__ == '__main__':
#     base_path = 'E:/experiments/Motion_model_fusionmask_raw_20220917/Motion_model_fusionmask_raw/figs'
#     name = 'MFSR_Sony_0015_x1'
#     img_path = '{}/{}.png'.format(base_path, name)
#     pkl_path = '{}/{}.pkl'.format(base_path, name)
#
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # [4, H, W] / [3, H, W]
#     img = img / 16383.
#
#     with open(pkl_path, 'rb') as f:
#         meta_data = pickle.load(f)
#
#     vis = generate_processed_image_channel4(img, meta_data, return_np=True)
#
#     vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
#
#     cv2.imwrite('{}/{}_vis.png'.format(base_path, name), vis)

