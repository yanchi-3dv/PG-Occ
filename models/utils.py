import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from typing import NamedTuple, Optional
from torch import Tensor

import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

from mmcv.cnn.bricks import ConvTranspose3d, Conv3d
from pyquaternion import Quaternion
from mpl_toolkits.axes_grid1 import ImageGrid
from plyfile import PlyElement, PlyData


def conv3d_gn_relu(in_channels, out_channels, kernel_size=1, stride=1):
    return nn.Sequential(
        Conv3d(in_channels, out_channels, kernel_size, stride, bias=False),
        nn.GroupNorm(16, out_channels),
        nn.ReLU(inplace=True),
    )

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def deconv3d_gn_relu(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias=False),
        nn.GroupNorm(16, out_channels),
        nn.ReLU(inplace=True),
    )


def sparse2dense(indices, value, dense_shape, empty_value=0):
    B, N = indices.shape[:2]  # [B, N, 3]

    batch_index = torch.arange(B).unsqueeze(1).expand(B, N)
    dense = torch.ones([B] + dense_shape, device=value.device, dtype=value.dtype) * empty_value
    dense[batch_index, indices[..., 0], indices[..., 1], indices[..., 2]] = value
    
    mask = torch.zeros([B] + dense_shape, dtype=torch.bool, device=value.device)
    mask[batch_index, indices[..., 0], indices[..., 1], indices[..., 2]] = 1

    return dense, mask


@torch.no_grad()
def generate_grid(n_vox, interval):
    # Create voxel grid
    grid_range = [torch.arange(0, n_vox[axis], interval) for axis in range(3)]
    grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2], indexing='ij'))  # 3 dx dy dz
    grid = grid.cuda().view(3, -1).permute(1, 0)  # N, 3
    return grid[None]  # 1, N, 3


def batch_indexing(batched_data: torch.Tensor, batched_indices: torch.Tensor, layout='channel_first'):
    def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, C, N]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, C, I1, I2, ..., Im]
        """
        def product(arr):
            p = 1
            for i in arr:
                p *= i
            return p
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size, n_channels = batched_data.shape[:2]
        indices_shape = list(batched_indices.shape[1:])
        batched_indices = batched_indices.reshape([batch_size, 1, -1])
        batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
        result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
        result = result.view([batch_size, n_channels] + indices_shape)
        return result

    def batch_indexing_channel_last(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, N, C]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, I1, I2, ..., Im, C]
        """
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)  # [bs, I1, I2, ..., Im]
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]

    if layout == 'channel_first':
        return batch_indexing_channel_first(batched_data, batched_indices)
    elif layout == 'channel_last':
        return batch_indexing_channel_last(batched_data, batched_indices)
    else:
        raise ValueError


class GridMask(nn.Module):
    def __init__(self, ratio=0.5, prob=0.7):
        super(GridMask, self).__init__()
        self.ratio = ratio
        self.prob = prob

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        d = np.random.randint(2, h)
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.uint8)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(hh // d):
            s = d*i + st_h
            t = min(s + l, hh)
            mask[s:t, :] = 0

        for i in range(ww // d):
            s = d*i + st_w
            t = min(s + l, ww)
            mask[:, s:t] = 0

        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
        mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
        mask = 1 - mask
        mask = mask.expand_as(x)
        x = x * mask 

        return x.view(n, c, h, w)


def rotation_3d_in_axis(points, angles):
    assert points.shape[-1] == 3
    assert angles.shape[-1] == 1
    angles = angles[..., 0]

    n_points = points.shape[-2]
    input_dims = angles.shape

    if len(input_dims) > 1:
        points = points.reshape(-1, n_points, 3)
        angles = angles.reshape(-1)

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    rot_mat_T = torch.stack([
        rot_cos, rot_sin, zeros,
        -rot_sin, rot_cos, zeros,
        zeros, zeros, ones,
    ]).transpose(0, 1).reshape(-1, 3, 3)

    points = torch.bmm(points, rot_mat_T)

    if len(input_dims) > 1:
        points = points.reshape(*input_dims, n_points, 3)
    
    return points


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pad_multiple(inputs, img_metas, size_divisor=32):
    _, _, img_h, img_w = inputs.shape

    pad_h = 0 if img_h % size_divisor == 0 else size_divisor - (img_h % size_divisor)
    pad_w = 0 if img_w % size_divisor == 0 else size_divisor - (img_w % size_divisor)

    B = len(img_metas)
    N = len(img_metas[0]['ori_shape'])

    for b in range(B):
        img_metas[b]['img_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]
        img_metas[b]['pad_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]

    if pad_h == 0 and pad_w == 0:
        return inputs
    else:
        return F.pad(inputs, [0, pad_w, 0, pad_h], value=0)


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image = image / 255.0

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0

    h = h * 360.0
    v = v * 255.0

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / 360.0
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :] / 255.0

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)
    out = out * 255.0

    return out


class GpuPhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, imgs):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = imgs[:, [2, 1, 0], :, :]  # BGR to RGB

        contrast_modes = []
        for _ in range(imgs.shape[0]):
            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            contrast_modes.append(random.randint(2))

        for idx in range(imgs.shape[0]):
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                imgs[idx] += delta

            if contrast_modes[idx] == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

        # convert color from BGR to HSV
        imgs = rgb_to_hsv(imgs)

        for idx in range(imgs.shape[0]):
            # random saturation
            if random.randint(2):
                imgs[idx, 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            if random.randint(2):
                imgs[idx, 0] += random.uniform(-self.hue_delta, self.hue_delta)

        imgs[:, 0][imgs[:, 0] > 360] -= 360
        imgs[:, 0][imgs[:, 0] < 0] += 360

        # convert color from HSV to BGR
        imgs = hsv_to_rgb(imgs)

        for idx in range(imgs.shape[0]):
            # random contrast
            if contrast_modes[idx] == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

            # randomly swap channels
            if random.randint(2):
                imgs[idx] = imgs[idx, random.permutation(3)]

        imgs = imgs[:, [2, 1, 0], :, :]  # RGB to BGR

        return imgs


def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]


class GaussianPrediction(NamedTuple):
    means: Optional[torch.Tensor] = None  
    scales: Optional[torch.Tensor] = None
    rotations: Optional[torch.Tensor] = None  
    opacities: Optional[torch.Tensor] = None
    semantics: Optional[torch.Tensor] = None
    ovs: Optional[torch.Tensor] = None

class GaussianPredictionTools:
    def __init__(self):
        self.use_localaggprob = False
        self.num_classes = 18
        self.dataset_type = 'nuscenes'

    def get_grid_coords(self, dims, resolution):
        """
        :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
        :return coords_grid: is the center coords of voxels in the grid
        """

        g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
        g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
        g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

        # Obtaining the grid with coords...
        xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
        coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
        coords_grid = coords_grid.astype(np.float32)
        resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

        coords_grid = (coords_grid * resolution) + resolution / 2

        return coords_grid


    def prepare_gaussian_args(self, gaussians, gaussian_interval=None, with_empty = False, base_scale_interval = 0.08):
        opacities = gaussians[-1].semantics
        means = gaussians[-1].means
        origi_opa = gaussians[-1].opacities.unsqueeze(-1)
        rotations = gaussians[-1].rotations
        scales = gaussians[-1].scales
        
        bs, g, semantic_dim = opacities.shape

        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        else:
            semantic_dim = semantic_dim + 1

        if with_empty:         # True
            empty_mean = torch.tensor([[[0., 0., 2.2]]]).cuda().repeat(bs, 1, 1)
            empty_scale = torch.tensor([[[100., 100., 8.]]]).cuda().repeat(bs, 1, 1)
            empty_rot = torch.tensor([[[1., 0., 0., 0.]]]).cuda().repeat(bs, 1, 1)
            empty_opa = torch.tensor([[[1.]]]).cuda().repeat(bs, 1, 1)
            empty_sem = torch.zeros(bs, 1, semantic_dim).cuda()

            opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1], requires_grad=False)], dim=-1)

            # add a new gs
            means = torch.cat([means, empty_mean], dim=1)
            scales = torch.cat([scales, empty_scale], dim=1)
            rotations = torch.cat([rotations, empty_rot], dim=1)
            empty_sem[..., -1] += 0.000001
            opacities = torch.cat([opacities, empty_sem], dim=1)
            origi_opa = torch.cat([origi_opa, empty_opa], dim=1)
            g = g+1
        
        elif self.use_localaggprob:
            assert opacities.shape[-1] == self.num_classes - 1
            opacities = opacities.softmax(dim=-1)
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        epsilon = 1e-5
        CovInv = torch.linalg.inv(Cov + epsilon * torch.eye(3, dtype=Cov.dtype, device=Cov.device))
        return means, origi_opa, opacities, scales, CovInv

    def save_gaussian_topdown(self, save_dir, gaussian, name):

        # init_means = safe_sigmoid(anchor_init[:, :2]) * 100 - 50
        # means = [init_means] + [g.means[0, :, :2] for g in gaussian]
        means = [g.means[0, :, :2] for g in gaussian]

        # from bev coo x up, y left-> matplt coo x right, y up
        means = [g.means[0, :, [1, 0]] * torch.tensor([-1, 1]).cuda() for g in gaussian]

        plt.clf(); plt.cla()
        fig = plt.figure(figsize=(24., 16.))

        grid = ImageGrid(fig, 111,              # similar to subplot(111)
                        nrows_ncols=(1, 3),     # creates 2x2 grid of Axes
                        axes_pad=0.,            # pad between Axes in inch.
                        share_all=True
                        )
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])
        for ax, im in zip(grid, means):
            im = im.cpu()
            # Iterating over the grid returns the Axes.
            ax.scatter(im[:, 0], im[:, 1], s=0.1, marker='o')
        plt.savefig(os.path.join(save_dir, f"{name}.jpg"))
        plt.clf(); plt.cla()
    
    def save_occ(
            self,
            save_dir, 
            occ_pred, 
            name,
            sem=False,
            cap=2,
            dataset='nusc'
        ):
        if dataset == 'nusc':
            voxel_size = [0.4] * 3
            vox_origin = [-40.0, -40.0, -1.0]
            vmin, vmax = 0, 16
        elif dataset == 'kitti':
            voxel_size = [0.2] * 3
            vox_origin = [0.0, -25.6, -2.0]
            vmin, vmax = 1, 19
        elif dataset == 'kitti360':
            voxel_size = [0.2] * 3
            vox_origin = [0.0, -25.6, -2.0]
            vmin, vmax = 1, 18
        voxels = torch.from_numpy(occ_pred)
        voxels[0, 0, 0] = 1
        voxels[-1, -1, -1] = 1
        
        if not sem:
            voxels[..., (-cap):] = 0
            for z in range(voxels.shape[-1] - cap):
                mask = (voxels > 0)[..., z]
                voxels[..., z][mask] = z + 1 
    
        # Compute the voxels coordinates
        grid_coords = self.get_grid_coords(
            voxels.shape, voxel_size
        ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
        # Get the voxels inside FOV
        fov_grid_coords = grid_coords

        # Remove empty and unknown voxels
        if not sem:
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 100)
            ]
        else:
            if dataset == 'nusc':
                fov_voxels = fov_grid_coords[
                    (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 17)
                ]
            elif dataset == 'kitti360':
                fov_voxels = fov_grid_coords[
                    (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 19)
                ]
            else:
                fov_voxels = fov_grid_coords[
                    (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
                ]
        print(len(fov_voxels))
        
        figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
        # Draw occupied inside FOV voxels
        voxel_size = sum(voxel_size) / 3
        if not sem:
            plt_plot_fov = mlab.points3d(
                fov_voxels[:, 0],
                -fov_voxels[:, 1],
                fov_voxels[:, 2],
                fov_voxels[:, 3],
                colormap="jet",
                scale_factor=1.0 * voxel_size,
                mode="cube",
                opacity=1.0,
            )
        else:
            plt_plot_fov = mlab.points3d(
                fov_voxels[:, 0],
                -fov_voxels[:, 1],
                fov_voxels[:, 2],
                fov_voxels[:, 3],
                scale_factor=1.0 * voxel_size,
                mode="cube",
                opacity=1.0,
                vmin=vmin,
                vmax=vmax, # 16
            )

        plt_plot_fov.glyph.scale_mode = "scale_by_vector"

        if sem:
            if dataset == 'nusc':
                colors = np.array(
                    [
                        [  0,   0,   0, 255],       # others
                        [255, 120,  50, 255],       # barrier              orange
                        [255, 192, 203, 255],       # bicycle              pink
                        [255, 255,   0, 255],       # bus                  yellow
                        [  0, 150, 245, 255],       # car                  blue
                        [  0, 255, 255, 255],       # construction_vehicle cyan
                        [255, 127,   0, 255],       # motorcycle           dark orange
                        [255,   0,   0, 255],       # pedestrian           red
                        [255, 240, 150, 255],       # traffic_cone         light yellow
                        [135,  60,   0, 255],       # trailer              brown
                        [160,  32, 240, 255],       # truck                purple                
                        [255,   0, 255, 255],       # driveable_surface    dark pink
                        # [175,   0,  75, 255],       # other_flat           dark red
                        [139, 137, 137, 255],
                        [ 75,   0,  75, 255],       # sidewalk             dard purple
                        [150, 240,  80, 255],       # terrain              light green          
                        [230, 230, 250, 255],       # manmade              white
                        [  0, 175,   0, 255],       # vegetation           green
                        # [  0, 255, 127, 255],       # ego car              dark cyan
                        # [255,  99,  71, 255],       # ego car
                        # [  0, 191, 255, 255]        # ego car
                    ]
                ).astype(np.uint8)
            elif dataset == 'kitti360':
                colors = (get_kitti360_colormap()[1:, :] * 255).astype(np.uint8)
            else:
                colors = (get_kitti_colormap()[1:, :] * 255).astype(np.uint8)

            plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
        
        scene = figure.scene
        scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.azimuth(-5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(-5)
        scene.render()
        scene.camera.position = [-138.7379881436844, -0.008333206176756428, 99.5084646673331]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [104.37185230017721, 252.84608651497263]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-114.65804807470022, -0.008333206176756668, 82.48137575398867]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [75.17498702830105, 222.91192666552377]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-94.75727115818437, -0.008333206176756867, 68.40940144543957]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [51.04534630774225, 198.1729515833347]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6463156430702276, -6.454925414290924e-18, 0.7630701733934554]
        scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6463156430702277, -6.4549254142909245e-18, 0.7630701733934555]
        scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.elevation(-5)
        mlab.pitch(-8)
        mlab.move(up=15)
        scene.camera.orthogonalize_view_up()
        scene.render()

        # scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
        # scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
        # scene.camera.view_angle = 40.0
        # scene.camera.view_up = [0.0, 0.0, 1.0]
        # scene.camera.clipping_range = [0.01, 300.]
        # scene.camera.compute_view_plane_normal()
        # scene.render()

        filepath = os.path.join(save_dir, f'{name}.png')
        if offscreen:
            mlab.savefig(filepath)
        else:
            mlab.show()
        mlab.close()
        
    def get_nuscenes_colormap(self):
        colors = np.array(
            [
                [  0,   0,   0, 255],       # others
                [255, 120,  50, 255],       # barrier              orange
                [255, 192, 203, 255],       # bicycle              pink
                [255, 255,   0, 255],       # bus                  yellow
                [  0, 150, 245, 255],       # car                  blue
                [  0, 255, 255, 255],       # construction_vehicle cyan
                [255, 127,   0, 255],       # motorcycle           dark orange
                [255,   0,   0, 255],       # pedestrian           red
                [255, 240, 150, 255],       # traffic_cone         light yellow
                [135,  60,   0, 255],       # trailer              brown
                [160,  32, 240, 255],       # truck                purple                
                [255,   0, 255, 255],       # driveable_surface    dark pink
                # [175,   0,  75, 255],     # other_flat           dark red
                [139, 137, 137, 255],       
                [ 75,   0,  75, 255],       # sidewalk             dard purple
                [150, 240,  80, 255],       # terrain              light green          
                [230, 230, 250, 255],       # manmade              white
                [  0, 175,   0, 255],       # vegetation           green
                # [  0, 255, 127, 255],     # ego car              dark cyan
                # [255,  99,  71, 255],     # ego car
                # [  0, 191, 255, 255]      # ego car
                [0, 0, 0, 255]              # empty
            ]
        ).astype(np.float32) / 255.
        
        return colors

    def save_gaussian(self, save_dir, gaussian, name, scalar=1, ignore_opa=False, filter_zsize=False):

        empty_label = 17
        sem_cmap = self.get_nuscenes_colormap()

        torch.save(gaussian, os.path.join(save_dir, f'{name}_attr.pth'))

        means = gaussian.means[0].detach().cpu().numpy() # g, 3
        scales = gaussian.scales[0].detach().cpu().numpy() # g, 3
        rotations = gaussian.rotations[0].detach().cpu().numpy() # g, 4
        opas = gaussian.opacities[0]
        if opas.numel() == 0:
            opas = torch.ones_like(gaussian.means[0][..., :1])
        opas = opas.squeeze().detach().cpu().numpy() # g
        # sems = torch.zeros_like(gaussian.means[0]).cpu().numpy() # g, 18
        # sems = gaussian.semantics[0].detach().cpu().numpy() # g, 18
        # pred = np.argmax(sems, axis=-1)
        pred = gaussian.semantics[0].detach().cpu().numpy() # g, 18

        ignore_opa = True
        if ignore_opa:  # False
            opas[:] = 1.
            mask = (pred != empty_label)
        else:
            mask = (pred != empty_label) & (opas > 0.5)

        if filter_zsize:
            zdist, zbins = np.histogram(means[:, 2], bins=100)
            zidx = np.argsort(zdist)[::-1]
            for idx in zidx[:10]:
                binl = zbins[idx]
                binr = zbins[idx + 1]
                zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
                mask = mask & zmsk
            
            z_small_mask = scales[:, 2] > 0.1
            mask = z_small_mask & mask

        means = means[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opas = opas[mask]
        pred = pred[mask]

        # number of ellipsoids 
        ellipNumber = means.shape[0]

        # set colour map so each ellipsoid as a unique colour
        norm = colors.Normalize(vmin=-1.0, vmax=5.4)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        fig = plt.figure(figsize=(9, 9), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=46, azim=-180)

        # compute each and plot each ellipsoid iteratively
        # border = np.array([
        #     [-50.0, -50.0, 0.0],
        #     [-50.0, 50.0, 0.0],
        #     [50.0, -50.0, 0.0],
        #     [50.0, 50.0, 0.0],
        # ])
        border = np.array([
            [-40.0, -40.0, 0.0],
            [-40.0, 40.0, 0.0],
            [40.0, -40.0, 0.0],
            [40.0, 40.0, 0.0],
        ])
        ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:], 
            rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

        from tqdm import tqdm
        for indx in tqdm(range(0, ellipNumber, 1), desc="Processing ellipsoids"):
            center = means[indx]
            radii = scales[indx] * scalar
            rot_matrix = rotations[indx]
            rot_matrix = Quaternion(rot_matrix).rotation_matrix.T

            # calculate cartesian coordinates for the ellipsoid surface
            u = np.linspace(0.0, 2.0 * np.pi, 10)
            v = np.linspace(0.0, np.pi, 10)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

            xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
            xyz = rot_matrix[None, None, ...] @ xyz[..., None]
            xyz = np.squeeze(xyz, axis=-1)

            xyz = xyz + center[None, None, ...]

            ax.plot_surface(
                xyz[..., 0], xyz[..., 1], xyz[..., 2], 
                rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

            # ax.plot_surface(
            #     xyz[..., 0], xyz[..., 1], xyz[..., 2], 
            #     rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)
        
        # plt.axis("equal")
        
        plt.axis("auto")
        plt.gca().set_box_aspect([1, 1, 1])
        ax.grid(False)
        ax.set_axis_off()

        filepath = os.path.join(save_dir, f'{name}.png')
        plt.savefig(filepath)

        plt.cla()
        plt.clf()
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        # for i in range(3):
        #     l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, ply_vis_dir, gau_pred, name):
        xyz = gau_pred.means.squeeze(0).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = gau_pred.opacities.squeeze(0).unsqueeze(-1).detach().cpu().numpy()
        scales = gau_pred.scales.squeeze(0).detach().cpu().numpy()
        rotations = gau_pred.rotations.squeeze(0).detach().cpu().numpy()

        if hasattr(gau_pred, 'semantics') and gau_pred.semantics is not None:
            semantics = gau_pred.semantics.squeeze(0).detach().cpu().numpy()
            # semantics = semantics.argmax(axis=-1, keepdims=True) 
            colors = self.get_nuscenes_colormap()[semantics][..., :3]
        else:
            colors = np.zeros_like(xyz)
                
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, colors, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(ply_vis_dir, name))
        
        print(f"PLY file saved to: {os.path.join(ply_vis_dir, name)}")


from functools import reduce

import numpy as np
import torch
from pyquaternion import Quaternion
from torch.cuda.amp import autocast


def cumprod(xs):
    return reduce(lambda x, y: x * y, xs)


def nlc_to_nchw(x, shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    B, L, C = x.shape
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    return x.flatten(2).transpose(1, 2).contiguous()


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([
        torch.tensor(feat.shape[2:], device=feat_flatten.device)
        for feat in feats
    ])
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))


def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [*grid_shape, len(grid_shape)]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= val
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(*shape_).expand(*grid_shape)
        grid.append(g)
    return torch.stack(grid, dim=-1)


def cam2world(points, cam2img, cam2ego, img_aug_mat=None):
    if img_aug_mat is not None:
        post_rots = img_aug_mat[..., :3, :3]
        post_trans = img_aug_mat[..., :3, 3]
        points = points - post_trans.unsqueeze(-2)
        points = (torch.inverse(post_rots).unsqueeze(2)
                  @ points.unsqueeze(-1)).squeeze(-1)

    cam2img = cam2img[..., :3, :3]
    with autocast(enabled=False):
        combine = cam2ego[..., :3, :3] @ torch.inverse(cam2img)
        points = points.float()
        points = torch.cat(
            [points[..., :2] * points[..., 2:3], points[..., 2:3]], dim=-1)
        points = combine.unsqueeze(2) @ points.unsqueeze(-1)
    points = points.squeeze(-1) + cam2ego[..., None, :3, 3]
    return points


def world2cam(points, cam2img, cam2ego, img_aug_mat=None, eps=1e-6):
    points = points - cam2ego[..., None, :3, 3]
    points = torch.inverse(cam2ego[..., None, :3, :3]) @ points.unsqueeze(-1)
    points = (cam2img[..., None, :3, :3] @ points).squeeze(-1)
    points = points / points[..., 2:3].clamp(eps)  # NOTE
    if img_aug_mat is not None:
        points = img_aug_mat[..., None, :3, :3] @ points.unsqueeze(-1)
        points = points.squeeze(-1) + img_aug_mat[..., None, :3, 3]
    return points[..., :2]


def rotmat_to_quat(rot_matrices):
    inputs = rot_matrices
    rot_matrices = rot_matrices.cpu().numpy()
    quats = []
    for rot in rot_matrices:
        while not np.allclose(rot @ rot.T, np.eye(3)):
            U, _, V = np.linalg.svd(rot)
            rot = U @ V
        quats.append(Quaternion(matrix=rot).elements)
    return torch.from_numpy(np.stack(quats)).to(inputs)


def quat_to_rotmat(quats):
    q = quats / torch.sqrt((quats**2).sum(dim=-1, keepdim=True))
    r, x, y, z = [i.squeeze(-1) for i in q.split(1, dim=-1)]

    R = torch.zeros((*r.shape, 3, 3)).to(r)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def get_covariance(s, r):
    L = torch.zeros((*s.shape[:2], 3, 3)).to(s)
    for i in range(s.size(-1)):
        L[..., i, i] = s[..., i]

    L = r @ L
    covariance = L @ L.mT
    return covariance


def unbatched_forward(func):

    def wrapper(*args, **kwargs):
        bs = None
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if bs is None:
                    bs = arg.size(0)
                else:
                    assert bs == arg.size(0)

        outputs = []
        for i in range(bs):
            output = func(
                *[
                    arg[i] if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ], **{
                    k: v[i] if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                })
            outputs.append(output)

        if isinstance(outputs[0], tuple):
            return tuple([
                torch.stack([out[i] for out in outputs])
                for i in range(len(outputs[0]))
            ])
        else:
            return torch.stack(outputs)

    return wrapper


def apply_to_items(func, iterable):
    if isinstance(iterable, list):
        return [func(i) for i in iterable]
    elif isinstance(iterable, dict):
        return {k: func(v) for k, v in iterable.items()}

OCC3D_CATEGORIES = (
    ['barrier'],
    ['bicycle'],
    ['bus'],
    ['car'],
    ['construction vehicle'],
    ['motorcycle'],
    ['person'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['sidewalk'],
    ['terrain', 'grass'],
    ['building', 'wall', 'fence', 'pole', 'sign'],
    ['vegetation'],
    ['sky'],
)  # `sum(OCC3D_CATEGORIES, [])` if you need to flatten the list.

class DumpConfig:
    def __init__(self):
        self.enabled = False
        self.out_dir = 'outputs'
        self.stage_count = 0
        self.frame_count = 0


DUMP = DumpConfig()
