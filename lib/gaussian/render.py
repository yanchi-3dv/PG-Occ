
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from gsplat import rasterization

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_depth_losses(render_depths, depth_gt):
    """Compute depth metrics, to allow monitoring during training
    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    depth_gt = depth_gt.clone()
    depth_gt = depth_gt.permute(1, 0, 2, 3)

    losses = {}
    depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

    mask = depth_gt > 0

    _, _, H, W = depth_gt.shape

    depth_pred = render_depths.detach()
    depth_pred = torch.clamp(F.interpolate(
        depth_pred, [H, W], mode="bilinear", align_corners=False), 1e-3, 80)

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]

    # if 'cam_T_cam' not in inputs:
        # depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)

    for i, metric in enumerate(depth_metric_names):
        losses[metric] = np.array(depth_errors[i].cpu().item())
    
    print(losses)

    return

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def camera_intrinsic_fov(intrinsic):
    w, h = intrinsic[0][2]*2, intrinsic[1][2]*2
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    
    # Go
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    return fov_x, fov_y

#! use in our ov task
def batch_splatting_render(pc, w2c, Ks, selected_numbers=None, render_conf=None, inference=False):
    assert pc.means.shape[0] == 1

    means = pc.means.squeeze(0).float()
    quats = pc.rotations.squeeze(0).float()
    scales = pc.scales.squeeze(0).float()
    opacities = pc.opacities.squeeze(0).float()
    
    Ks = Ks[:, :3, :3]

    width, height = render_conf['render_w'], render_conf['render_h']
    
    if pc.ovs is not None:
        # get ov feature
        semantics = pc.ovs.squeeze(0).float()
    else:
        semantics = torch.zeros_like(means)

    ov_features = []

    if inference:
        
        render_results, alphas, meta = rasterization(
            means, quats, scales, opacities, semantics, w2c, Ks, width, height, packed=False, sparse_grad=False, render_mode="ED",
        )

        return {"depth": render_results[..., -1:], "alphas": alphas}

    render_results, alphas, meta = rasterization(
        means, quats, scales, opacities, semantics, w2c, Ks, width, height, packed=False, sparse_grad=False, render_mode="RGB+ED"
    )

    if pc.ovs is not None:
        ov_features = render_results[..., :-1]            # [6, 180, 320, 1]
    else:
        ov_features = None

    depth = render_results[..., -1:]

    return {"depth": depth, 
            "ov_feature": ov_features,
            "alphas": alphas,
            "meta": meta}

def prepare_gs_attribute(img_metas):
    cam2ego = np.array(img_metas[0]['cam2ego'])
    render_k = np.array(img_metas[0]['render_k'])
    
    C2W = torch.tensor(cam2ego).float().cuda()   # [6, 4, 4]
    W2C = torch.inverse(C2W)
    render_k = torch.tensor(render_k[0:6]).float().cuda()

    return render_k, C2W, W2C

def save_images(images, title, filename):
    num_images = images.shape[0]
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        img = images[i].permute(1, 2, 0)
        plt.imshow(img.clamp(0, 1))
        plt.axis('off')

    plt.suptitle(title)
    plt.savefig(filename)
    plt.close()



class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask, opt):

        if opt['loss'] == 'silog':
            d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
            return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

        elif opt['loss'] == 'l1':
            return F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

        elif opt['loss'] == 'rl1':

            depth_est = (1/depth_est) * opt['max_depth']
            depth_gt = (1/depth_gt) * opt['max_depth']
            # depth_est = 1 / depth_est
            # depth_gt = 1 / depth_gt
            return F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

        elif opt['loss'] == 'sml1':
            return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

        else:
            print('please define the loss')
            exit()


def get_depth_loss(depth_render, depth, mask):
    
    silog_criterion = silog_loss(variance_focus=0.85)
    singel_scale_total_loss = 0

    # width_ori, height_ori = 1600, 900

    if True:
        opt = {
            'loss': 'silog',
            'max_depth': 80.0,
            'variance_focus': 0.1
        }

        silog_loss_result = silog_criterion.forward(depth_render, depth, mask.to(torch.bool), opt)
        l1_loss_result = F.l1_loss(depth_render[mask], depth[mask], size_average=True)

        singel_scale_total_loss += (silog_loss_result * 0.15 + l1_loss_result * 0.85)

    return singel_scale_total_loss

def get_gt_loss(disp, depth_gt, mask):
    
    silog_criterion = silog_loss(variance_focus=0.85)
    singel_scale_total_loss = 0

    width_ori, height_ori = 1600, 900

    if True:
        depth_pred = F.interpolate(disp, size=[height_ori, width_ori], mode="bilinear", align_corners=False).squeeze(1)

        opt = {
            'loss': 'silog',
            'max_depth': 80.0,
            'variance_focus': 0.1
        }
        no_aug_loss = silog_criterion.forward(depth_pred, depth_gt, mask.to(torch.bool), opt)

        singel_scale_total_loss += no_aug_loss

    return singel_scale_total_loss

if __name__ == "__main__":
    pass
