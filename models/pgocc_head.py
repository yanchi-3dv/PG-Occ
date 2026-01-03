import time
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.runner import auto_fp16
from mmdet.models.utils import build_transformer
from lib.gaussian.render import (
    prepare_gs_attribute, 
    batch_splatting_render, 
    get_gt_loss, 
    get_depth_loss
)
from .loss_utils import BackprojectDepth, Project3D, calc_time_warping_loss
from .utils import OCC3D_CATEGORIES

def get_nuScenes_label_name(label_mapping):
    """
    Load nuScenes label names and learning map from yaml config.
    
    Args:
        label_mapping: Path to yaml file containing label mappings
    
    Returns:
        nuScenes_label_name: Dictionary mapping label IDs to names
        learning_map: Dictionary mapping original labels to learned labels
    """
    with open(label_mapping, 'r') as stream:
        nuscenes_yaml = yaml.safe_load(stream)
    
    nuscenes_label_name = {}
    for i in sorted(list(nuscenes_yaml['learning_map'].keys()))[::-1]:
        val = nuscenes_yaml['learning_map'][i]
        nuscenes_label_name[val] = nuscenes_yaml['labels_16'][val]
    
    return nuscenes_label_name, nuscenes_yaml['learning_map']

@HEADS.register_module()
class PGOccHead(nn.Module):
    """
    PG-Occ Head for 3D scene understanding.
    """
    def __init__(self,
                 transformer=None,
                 class_names=None,
                 embed_dims=None,
                 occ_size=None,
                 pc_range=None,
                 loss_cfgs=None,
                 metric=[],
                 render_conf=None,
                 use_pca=False,
                 voxelizer=None,
                 text_prompt_paths=[''],
                 use_gt_depth=False,
                 loss_weights=None,
                 eval_thresholds=None,
                 **kwargs):
        """
        Initialize PGOccHead.
        
        Args:
            transformer: Transformer configuration
            class_names: List of class names for occupancy prediction
            embed_dims: Embedding dimensions
            occ_size: Size of occupancy grid
            pc_range: Point cloud range
            loss_cfgs: Loss function configurations
            metric: List of metrics to compute
            render_conf: Rendering configuration dictionary
            use_pca: Whether to use PCA for feature reduction
            voxelizer: Voxelizer module configuration
            text_prompt_paths: Paths to text prompt embeddings
            use_gt_depth: Whether to use ground truth depth for training
            loss_weights: Dictionary of loss weights
            eval_thresholds: Dictionary of evaluation thresholds
        """
        super(PGOccHead, self).__init__()
        
        # Basic configurations
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.pc_range = pc_range
        self.occ_size = occ_size
        self.embed_dims = embed_dims
        self.render_conf = render_conf
        self.use_pca = use_pca
        self.use_gt_depth = use_gt_depth

        # Loss weights configuration with default values
        default_loss_weights = dict(
            depth_warping=10.0,
            ov_mse=10.0,
            ov_cos=1.0,
            depth_foundation=0.5,
            depth_gt=0.5,
        )
        self.loss_weights = loss_weights if loss_weights is not None else default_loss_weights
        
        # Evaluation thresholds with default values
        default_eval_thresholds = dict(
            density_threshold=4e-2,
        )
        self.eval_thresholds = eval_thresholds if eval_thresholds is not None else default_eval_thresholds

        # Training/evaluation state
        self.cnt = 0
        self.training_iter = 0
        self.timestamp = int(time.time())
        
        # Build transformer
        self.transformer = build_transformer(transformer)
        self.metric = metric

        # Camera configuration
        self.cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        # Initialize 2d-3d projection modules
        num_cam = 6
        render_h = render_conf['render_h']
        render_w = render_conf['render_w']
        self.backproject_depth = BackprojectDepth(num_cam, render_h, render_w).cuda()
        self.project_3d = Project3D(num_cam, render_h, render_w).cuda()

        # Get distributed training rank
        try:
            self.rank = torch.distributed.get_rank()
            print(f"Current rank: {self.rank}")
        except:
            self.rank = 0
            print(f"Current rank: {self.rank}")
    
        # Load text prompt embeddings for open-vocabulary
        for weights_path in text_prompt_paths:
            if os.path.exists(weights_path):
                self.text_proto_embeds = torch.load(
                    weights_path, map_location='cpu').cuda()
                break
        
        # Build voxelizer if provided
        if voxelizer is not None:
            self.voxelizer = HEADS.build(voxelizer)

    def init_weights(self):
        """Initialize weights of the transformer."""
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, depth):
        """
        Forward pass of PGOccHead.
        
        Args:
            mlvl_feats: Multi-level image features
            img_metas: Image meta information
            depth: Depth maps
        
        Returns:
            Dictionary containing Gaussian predictions
        """
        gau_preds = self.transformer(
            mlvl_feats,
            img_metas=img_metas,
            depth=depth
        )
        
        return {
            'gau_preds': gau_preds,
        }

    def loss(self, preds_dicts, img_metas, mask_camera=None, **kwargs):
        """
        Compute losses.
        
        Args:
            preds_dicts: Prediction dictionaries
            img_metas: Image meta information
            mask_camera: Camera mask (unused, kept for compatibility)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of losses
        """
        return self.loss_2d_single(preds_dicts, img_metas=img_metas, **kwargs)

    def loss_2d_single(self, preds_dicts, img_metas, **kwargs):
        """
        Compute 2D losses for depth warping and open-vocabulary features.
        
        Args:
            preds_dicts: Dictionary containing Gaussian predictions
            img_metas: Image meta information
            **kwargs: Additional data including text_vision, depth, etc.
        
        Returns:
            Dictionary of computed losses
        """
        loss_dict = {}
        gau_preds = preds_dicts['gau_preds']
        self.training_iter += 1

        # Prepare camera attributes
        K, C2W, W2C = prepare_gs_attribute(img_metas)

        # Preprocess text-vision features with PCA
        ov_tgt_feature = kwargs['text_vision'].clone().detach().permute(0, 1, 3, 4, 2)
        ov_tgt_feature_pca = ov_tgt_feature.flatten(2, 3)

        # Apply PCA to reduce dimensionality to 128
        pca_u, pca_s, pca_v = torch.pca_lowrank(
            ov_tgt_feature_pca.flatten(0, 2).double(), 
            q=128, 
            niter=4
        )

        ov_tgt_feature = ov_tgt_feature @ pca_v.to(ov_tgt_feature)

        # Reshape and interpolate to render resolution
        ov_tgt_feature = ov_tgt_feature.permute(0, 1, 4, 2, 3)
        B, C, D, H, W = ov_tgt_feature.shape
        ov_tgt_feature = ov_tgt_feature.reshape(B, C*D, H, W)
        ov_tgt_feature = F.interpolate(
            ov_tgt_feature, 
            size=(self.render_conf['render_h'], self.render_conf['render_w']), 
            mode='bilinear', 
            align_corners=False
        )
        ov_tgt_feature = ov_tgt_feature.reshape(
            B, C, D, self.render_conf['render_h'], self.render_conf['render_w']
        ).permute(0, 1, 3, 4, 2)

        # Iterate through Gaussian predictions at different levels
        for i, gaussian in enumerate(gau_preds):
            loss_dict_i = {}

            # Apply PCA to Gaussian open-vocabulary features
            if gaussian.ovs is not None:
                gaussian.ovs = gaussian.ovs @ pca_v.to(gaussian.ovs)

            # Render depth and features from Gaussians
            render_results = batch_splatting_render(
                gaussian, W2C, K, render_conf=self.render_conf)

            render_depths = render_results['depth']
            render_depths = render_depths.permute(0, 3, 1, 2)
            render_depths = render_depths.clamp(min=0.1, max=80.0)

            # Extract rendered open-vocabulary features
            if gaussian.ovs is not None:
                ov_feature = render_results['ov_feature'].unsqueeze(0)

            # Compute temporal depth warping loss
            t0_2_x_geo = kwargs['t0_2_x_geo']
            render_gt = kwargs['render_gt']
            loss_depth_warping = calc_time_warping_loss(
                render_depths[0:6], 
                t0_2_x_geo, 
                render_gt, 
                self.backproject_depth, 
                self.project_3d, 
                K
            )
            loss_dict_i['loss_depth_warping'] = loss_depth_warping * self.loss_weights['depth_warping']

            # Compute open-vocabulary feature losses
            if gaussian.ovs is not None:
                # MSE loss for feature matching
                loss_dict_i['loss_ov_mse'] = F.mse_loss(ov_feature, ov_tgt_feature) * self.loss_weights['ov_mse']

                # Cosine similarity loss for normalized features
                ov_feature_normed = ov_feature / ov_feature.norm(dim=-1, keepdim=True)
                ov_tgt_feature_normed = ov_tgt_feature / ov_tgt_feature.norm(dim=-1, keepdim=True)
                ov_feature_flat = ov_feature_normed.reshape(-1, D)
                ov_tgt_feature_flat = ov_tgt_feature_normed.reshape(-1, D)
                loss_dict_i['loss_ov_cos'] = (1.0 - torch.nanmean(
                    F.cosine_similarity(ov_feature_flat, ov_tgt_feature_flat))) * self.loss_weights['ov_cos']
            
            # Compute depth loss with foundation depth model
            depth_foundation_tgt = kwargs['depth'].clone().squeeze(0)
            mask = (depth_foundation_tgt > 0.1) & (depth_foundation_tgt < 51.2)
            mask.detach_()
            depth_loss = self.loss_weights['depth_foundation'] * get_depth_loss(
                render_depths, depth_foundation_tgt, mask)
            loss_dict_i['depth_loss'] = loss_dict_i.get('loss', 0) + depth_loss

            # Optional: compute loss with ground truth depth
            if self.use_gt_depth:
                depth_gt_training = kwargs['gt_depth'].clone().squeeze(0)
                mask = (depth_gt_training > 0.1) & (depth_gt_training < 80.0)
                mask.detach_()
                gt_loss = self.loss_weights['depth_gt'] * get_gt_loss(
                    render_depths, depth_gt_training, mask)
                loss_dict_i['gt_loss'] = loss_dict_i.get('loss', 0) + gt_loss

            # Add losses for this Gaussian level to main loss dict
            for k, v in loss_dict_i.items():
                loss_dict[f'{k}_{i}'] = v.float()

        return loss_dict

    def merge_probs(self, probs, categories):
        """
        Merge probabilities for categories that have multiple sub-classes.
        
        For categories with multiple sub-classes, takes the maximum probability.
        
        Args:
            probs: Probability tensor [..., num_classes]
            categories: List of category groups, each containing sub-class indices
        
        Returns:
            Merged probabilities [..., num_merged_classes]
        """
        merged_probs = []
        idx = 0
        for cats in categories:
            p = probs[..., idx:idx + len(cats)]
            idx += len(cats)
            # For multi-class categories, take max probability
            if len(cats) > 1:
                p = p.max(-1, keepdim=True).values
            merged_probs.append(p)
        return torch.cat(merged_probs, dim=-1)

    def merge_occ_pred(self, outs, img_metas, points_in_ego=None):
        """
        Merge Gaussian predictions into occupancy predictions.
        
        Converts Gaussian distributions to voxel occupancy predictions using
        voxelizer and open-vocabulary text embeddings.
        
        Args:
            outs: Output dictionary containing Gaussian predictions
            img_metas: Image meta information
            points_in_ego: Optional points in ego coordinate system
        
        Returns:
            Updated outputs with occupancy predictions
        """
        if hasattr(self, 'voxelizer'):
            # Use the third Gaussian prediction level (index 2) for evaluation
            eval_gaussian_index = 2
            gaussian = outs['gau_preds'][eval_gaussian_index]

            # Compute class similarity using text embeddings
            class_similarity = torch.einsum(
                'bnd,dm->bnm', 
                gaussian.ovs, 
                self.text_proto_embeds
            )

            # If specific points provided, voxelize at those locations only
            if points_in_ego is not None:
                density, grid_feats = self.voxelizer(
                    means3d=gaussian.means,
                    opacities=gaussian.opacities,
                    features=gaussian.ovs,
                    rotations=gaussian.rotations,
                    scales=gaussian.scales,
                    points_in_ego=points_in_ego,
                )
                return density, grid_feats

            # Voxelize Gaussians to grid
            density, grid_feats = self.voxelizer(
                means3d=gaussian.means,
                opacities=gaussian.opacities,
                features=class_similarity,
                rotations=gaussian.rotations,
                scales=gaussian.scales,
            )

            # Compute semantic predictions for Gaussians
            gaussian_probs = self.merge_probs(
                class_similarity.softmax(-1), OCC3D_CATEGORIES)
            gaussian_preds = gaussian_probs.argmax(-1)
            # Adjust class indices for nuScenes convention (skip index 11)
            gaussian_preds += (gaussian_preds > 10) * 1 + 1
            gaussian.semantics = gaussian_preds

            # Compute occupancy predictions from voxelized features
            probs = grid_feats.softmax(-1)
            probs = self.merge_probs(probs, OCC3D_CATEGORIES)
        
            preds = probs.argmax(-1)
            # Adjust class indices for nuScenes convention
            preds += (preds > 10) * 1 + 1
            # Apply density threshold: use class 17 (empty) for low density
            density_threshold = self.eval_thresholds['density_threshold']
            preds = torch.where(density.squeeze(-1) > density_threshold, preds, 17)
            occ_preds = preds
        
        outs['occ_preds'] = occ_preds.cpu().numpy()

        # Optionally render depths for evaluation
        if 'depth' in self.metric:
            self.cnt += 1
            K, C2W, W2C = prepare_gs_attribute(img_metas)
            
            for i, gaussian in enumerate(outs['gau_preds']):
                # Render depth from Gaussians
                render_results = batch_splatting_render(
                    gaussian, W2C, K, render_conf=self.render_conf)
                render_depths = render_results['depth']
                render_depths = render_depths.permute(0, 3, 1, 2)
                render_depths = render_depths.clamp(min=0.1, max=80.0)
                outs[f'render_depths_{i}'] = render_depths
        
        return outs
    