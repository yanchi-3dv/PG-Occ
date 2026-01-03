import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Global SSIM instance for efficiency
_SSIM_INSTANCE = None


def get_ssim():
    """Get or create a global SSIM instance"""
    global _SSIM_INSTANCE
    if _SSIM_INSTANCE is None:
        _SSIM_INSTANCE = SSIM().cuda()
    return _SSIM_INSTANCE


def calc_time_warping_loss(depths, t0_2_tn, render_gt, backproject_depth, project_3d, k):
    """
    Calculate temporal warping loss for depth estimation.
    
    Args:
        depths: Predicted depth maps
        t0_2_tn: Transformation matrices from time t0 to tn
        render_gt: Ground truth rendered images
        backproject_depth: Function to backproject depth to 3D points
        project_3d: Function to project 3D points to 2D
        k: Camera intrinsic matrices
    
    Returns:
        loss: Computed warping loss
    """
    reprojection_losses = []
    identity_reprojection_losses = []

    # Normalize and prepare images
    render_gt = render_gt[0].permute(0, 3, 1, 2) / 255.0  
    t0_img = render_gt[0:6]
    tn_img = render_gt[6:3*6]
    k = k[0:6].float()
    inv_k = torch.inverse(k).float()
    
    # Backproject depth to camera points
    depths = depths.float()
    cam_points = backproject_depth(depths, inv_k)

    # Iterate through past frames
    num_past_frames = int(len(tn_img) / 6)
    for past_i in range(num_past_frames):
        # Project to pixel coordinates
        pix_coords, _ = project_3d(
            cam_points, k, t0_2_tn[0][6*past_i:6*(past_i+1)].float())

        # Warp image using grid sample
        warped_img = F.grid_sample(
            tn_img[past_i*6:(past_i+1)*6],
            pix_coords,
            padding_mode="border", 
            align_corners=True)

        # Compute reprojection losses
        reprojection_loss = compute_reprojection_loss(warped_img, t0_img)
        reprojection_losses.append(reprojection_loss)
        
        identity_reprojection_loss = compute_reprojection_loss(
            tn_img[past_i*6:(past_i+1)*6], t0_img)
        identity_reprojection_losses.append(identity_reprojection_loss)
    
    # Combine losses
    reprojection_losses = torch.cat(reprojection_losses, dim=0)
    identity_reprojection_losses = torch.cat(identity_reprojection_losses, dim=0)

    # Add small noise to break ties
    identity_reprojection_losses += torch.randn(
        identity_reprojection_losses.shape, device=identity_reprojection_losses.device) * 0.00001
    
    # Select minimum loss (auto-masking)
    combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)
    to_optimise, _ = torch.min(combined, dim=1)

    loss = to_optimise.mean()
    return loss

class BackprojectDepth(nn.Module):
    """
    Layer to transform a depth image into a point cloud.
    
    This module backprojects 2D depth maps to 3D camera coordinates using
    camera intrinsics.
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # Create pixel coordinate grid
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False)

        # Create homogeneous coordinates
        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False)

        # Flatten and stack pixel coordinates
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        """
        Backproject depth to 3D camera points.
        
        Args:
            depth: Depth map [B, H, W]
            inv_K: Inverse camera intrinsics [B, 4, 4]
        
        Returns:
            cam_points: 3D points in camera coordinates [B, 4, H*W]
        """
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points

class SSIM(nn.Module):
    """
    Layer to compute the SSIM (Structural Similarity Index) loss between image pairs.
    
    SSIM is a perceptual metric that quantifies image quality degradation caused
    by processing such as data compression or transmission.
    """
    def __init__(self):
        super(SSIM, self).__init__()
        # Average pooling layers for computing statistics
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        # Constants for stability
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        """
        Compute SSIM loss between two images.
        
        Args:
            x: First image tensor
            y: Second image tensor
        
        Returns:
            SSIM loss (lower is better)
        """
        x = self.refl(x)
        y = self.refl(y)

        # Compute means
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        # Compute variances and covariance
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        # Compute SSIM
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def compute_reprojection_loss(pred, target):
    """
    Compute reprojection loss between predicted and target images.
    
    Uses a combination of SSIM and L1 loss for robust photometric consistency.
    
    Args:
        pred: Predicted image tensor [B, C, H, W]
        target: Target image tensor [B, C, H, W]
    
    Returns:
        reprojection_loss: Combined loss [B, 1, H, W]
    """
    # Use global SSIM instance for efficiency
    ssim = get_ssim()
    
    # Compute L1 loss
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    
    # Compute SSIM loss
    ssim_loss = ssim(pred, target).mean(1, True)
    
    # Combine losses: 85% SSIM + 15% L1
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

class Project3D(nn.Module):
    """
    Layer to project 3D points into 2D image coordinates.
    
    Projects 3D points using camera intrinsics K and extrinsics T,
    then normalizes to grid coordinates for grid_sample.
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        """
        Project 3D points to 2D pixel coordinates.
        
        Args:
            points: 3D points in camera coordinates [B, 4, N]
            K: Camera intrinsics [B, 4, 4]
            T: Camera extrinsics (transformation matrix) [B, 4, 4]
        
        Returns:
            pix_coords: Normalized pixel coordinates for grid_sample [-1, 1] [B, H, W, 2]
            pix_coords_unnorm: Unnormalized pixel coordinates [B, H, W, 2]
        """
        # Project points: P = K * T
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        # Perspective division
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        
        # Keep unnormalized coordinates for debugging
        pix_coords_unnorm = pix_coords.clone()

        # Normalize to [-1, 1] for grid_sample
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        
        return pix_coords, pix_coords_unnorm
