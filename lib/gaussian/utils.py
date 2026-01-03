from typing import NamedTuple, Optional
import torch
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class GaussianPrediction:
    means: Optional[torch.Tensor] = None  
    scales: Optional[torch.Tensor] = None  
    rotations: Optional[torch.Tensor] = None  
    opacities: Optional[torch.Tensor] = None
    colors: Optional[torch.Tensor] = None
    ovs: Optional[torch.Tensor] = None
    semantics: Optional[torch.Tensor] = None

def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()
    
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    
    # Create boolean mask for invalid values
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth))).astype(bool)
    
    if mask is not None:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        invalid_mask = np.logical_or(invalid_mask, np.logical_not(mask)).astype(bool)
    
    if depth_min is None:
        depth_min = np.percentile(depth[~invalid_mask], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[~invalid_mask], 95)
    
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    
    # Ensure the array is 2D for cv2.applyColorMap
    if depth_scaled_uint8.ndim != 2:
        if depth_scaled_uint8.ndim == 3 and depth_scaled_uint8.shape[-1] == 1:
            depth_scaled_uint8 = depth_scaled_uint8.squeeze(-1)
        else:
            raise ValueError(f"Expected 2D depth array, got shape {depth_scaled_uint8.shape}")
    
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))

    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3