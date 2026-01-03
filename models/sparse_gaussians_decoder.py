import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import FFN
from .sparsebev_transformer import SparseBEVSelfAttention, SparseBEVSampling, AdaptiveMixing
from .bbox.utils import encode_bbox
from lib.gaussian.utils import GaussianPrediction
from .loss_utils import BackprojectDepth
from pointops import farthest_point_sampling
from lib.gaussian.render import prepare_gs_attribute, batch_splatting_render

def index2point(coords, pc_range, voxel_size):
    """
    coords: [B, N, 3], int
    pc_range: [-40, -40, -1.0, 40, 40, 5.4]
    voxel_size: float
    """
    coords = coords * voxel_size
    coords = coords + torch.tensor(pc_range[:3], device=coords.device)
    return coords

def point2bbox(coords, box_size, quaternion_params=None):
    """
    coords: [B, N, 3], float
    box_size: float
    """
    if isinstance(box_size, float) or isinstance(box_size, int):
        wlh = torch.ones_like(coords.float()) * box_size
    elif box_size.dim() == 3:
        wlh = box_size
    else:
        raise ValueError(f"Invalid box_size dimension: {box_size.dim()}")
    if quaternion_params is not None:
        bboxes = torch.cat([coords, wlh, quaternion_params], dim=-1)  # [B, N, 10]
    else:
        bboxes = torch.cat([coords, wlh], dim=-1)  # [B, N, 6]
    return bboxes

class SparseGaussiansDecoder(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 layers_scale=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 topk_training=None,
                 topk_testing=None,
                 pc_range=[],
                 gaussian_scale_range=None,
                 render_conf=None,
                 restrict_xyz=True,
                 num_queries=[2000],
                 use_anisotropy_encoding=True):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.layers_scale = layers_scale
        self.pc_range = pc_range
        self.voxel_dim = [200, 200, 16]
        self.topk_training = topk_training
        self.topk_testing = topk_testing
        self.use_anisotropy_encoding = use_anisotropy_encoding

        self.decoder_layers = nn.ModuleList()
        
        self.render_conf = render_conf

        self.gau_pred_heads = nn.ModuleList()
        self.ov_heads = nn.ModuleList()

        self.num_queries = num_queries
        self.query_embeds = nn.Embedding(self.num_queries[0] + self.num_queries[1] + self.num_queries[2], embed_dims)
        
        num_cam = 6
        h = self.render_conf['render_h']
        w = self.render_conf['render_w']
        self.backproject_depth = BackprojectDepth(num_cam, h, w).cuda()

        if restrict_xyz:
            unit_xyz = [4.0, 4.0, 0.32]
            unit_prob = [unit_xyz[i] / (self.pc_range[i + 3] - self.pc_range[i]) for i in range(3)]
            unit_prob = [8 * unit_prob[i] for i in range(3)]
            self.unit_sigmoid = unit_prob
            
        self.layers_scales = ['coarse', 'medium', 'fine']
        for i, _ in enumerate(self.layers_scales):  # [0, 1, 2]
            
            self.decoder_layers.append(SparseGaussiansDecoderLayer(
                embed_dims=embed_dims,
                num_frames=num_frames,
                num_points=num_points,
                num_groups=num_groups,
                num_levels=num_levels,
                pc_range=self.pc_range,
                self_attn=True,
                past_queries=sum(self.num_queries[:i]),
                use_anisotropy_encoding=use_anisotropy_encoding
            ))

            self.gau_pred_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims * 4, 11)
            ))

            if self.render_conf['use_ov']:
                self.ov_heads.append(nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(embed_dims, embed_dims * 4),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Sequential(
                        nn.Linear(embed_dims * 4, embed_dims * 4),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Sequential(
                        nn.Linear(embed_dims * 4, 512),
                    )
                ]))
    
        unit_quaternion = torch.zeros(1, 1, 4, device=torch.device('cuda'), dtype=torch.float32)
        unit_quaternion[..., 0] = 1.0
        self.unit_quaternion = unit_quaternion

    @torch.no_grad()
    def init_weights(self):
        for i in range(len(self.decoder_layers)):
            self.decoder_layers[i].init_weights()

    def query_2_gaussian(self, gau_pred, range=[0.1, 6.4]):
        if torch.isnan(gau_pred).any():
            print("Warning: NaN detected in gau_pred input")
            gau_pred = torch.nan_to_num(gau_pred, nan=0.0)
        
        gau_xyz_sigmoid = 2 * torch.nn.functional.sigmoid(gau_pred[...,0:3]) - 1
        gau_xyz_delta = torch.stack(
            [
                gau_xyz_sigmoid[...,0] * self.unit_sigmoid[0],
                gau_xyz_sigmoid[...,1] * self.unit_sigmoid[1],
                gau_xyz_sigmoid[...,2] * self.unit_sigmoid[2],
            ],
            dim=-1
        )
        
        if torch.isnan(gau_xyz_delta).any():
            print("Warning: NaN detected in gau_xyz_delta")
            gau_xyz_delta = torch.nan_to_num(gau_xyz_delta, nan=0.0)
        
        gau_rots = torch.nn.functional.normalize(gau_pred[...,3:7], dim=-1)
        
        if torch.isnan(gau_rots).any():
            print("Warning: NaN detected in gau_rots")
            gau_rots = torch.nan_to_num(gau_rots, nan=0.0)
        
        gau_scales = (torch.nn.functional.sigmoid(gau_pred[...,7:10])*(range[1]-range[0])) + range[0]
        
        if torch.isnan(gau_scales).any():
            print("Warning: NaN detected in gau_scales")
            gau_scales = torch.nan_to_num(gau_scales, nan=range[0])
        
        gau_opacities = torch.nn.functional.sigmoid(gau_pred[...,10:11]).squeeze(-1)
        
        if torch.isnan(gau_opacities).any():
            print("Warning: NaN detected in gau_opacities")
            gau_opacities = torch.nan_to_num(gau_opacities, nan=0.5)
        
        result = dict(
            delta_xyz=gau_xyz_delta,
            gau_rots=gau_rots,
            gau_scales=gau_scales,
            gau_opacities=gau_opacities
        )
        return result

    def forward(self, mlvl_feats, img_metas, depth=None):

        B = len(img_metas)
        gau_preds = []

        for i, feat in enumerate(mlvl_feats):
            if torch.isnan(feat).any():
                print(f"Warning: NaN detected in mlvl_feats[{i}]")
                mlvl_feats[i] = torch.nan_to_num(feat, nan=0.0)

        query_feat = self.query_embeds.weight.unsqueeze(0)
        
        if torch.isnan(query_feat).any():
            print("Warning: NaN detected in query_feat")
            query_feat = torch.nan_to_num(query_feat, nan=0.0)

        if depth is not None:

            depth = depth[0]
            depth = depth.clamp(max=51.2)

            mask = torch.logical_and(depth > 0, depth < 80).squeeze(1)
            render_k, cam2ego, W2C = prepare_gs_attribute(img_metas)

            inv_k = torch.inverse(render_k)
            cam_points = self.backproject_depth(depth, inv_k)
            
            if torch.isnan(cam_points).any():
                print("Warning: NaN detected in cam_points")
                cam_points = torch.nan_to_num(cam_points, nan=0.0)

            ego_points = cam2ego @ cam_points
            ego_points = ego_points[...,0:3, :]
            ego_points = ego_points.reshape(6, 3, self.render_conf['render_h'], self.render_conf['render_w']).permute(0, 2, 3, 1)

            if torch.isnan(ego_points).any():
                print("Warning: NaN detected in ego_points")
                ego_points = torch.nan_to_num(ego_points, nan=0.0)

            roi_mask = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(ego_points[..., 0] > self.pc_range[0], ego_points[..., 0] < self.pc_range[3]), ego_points[..., 1] > self.pc_range[1]), ego_points[..., 1] < self.pc_range[4]), ego_points[..., 2] > self.pc_range[2]), ego_points[..., 2] < self.pc_range[5])

            ego_points_coarse = ego_points[mask]
            
            if ego_points_coarse.numel() == 0:
                print("Warning: ego_points_coarse is empty, using fallback coordinates")
                ego_points_coarse = torch.randn(1000, 3, device=ego_points.device) * 10.0
            
            if torch.isnan(ego_points_coarse).any():
                print("Warning: NaN detected in ego_points_coarse")
                ego_points_coarse = torch.nan_to_num(ego_points_coarse, nan=0.0)

            try:
                selected_ego_points = farthest_point_sampling(
                    ego_points_coarse, 
                    torch.tensor([ego_points_coarse.shape[0]], device=ego_points_coarse.device, dtype=torch.int),
                    torch.tensor([self.num_queries[0]+self.num_queries[1]+self.num_queries[2]], device=ego_points_coarse.device, dtype=torch.int)
                )
                selected_ego_points_coarse = selected_ego_points[:self.num_queries[0]]
                query_coord = ego_points_coarse[selected_ego_points_coarse].reshape(B, -1, 3)
            except Exception as e:
                print(f"Warning: FPS failed: {e}, using random sampling")
                indices = torch.randperm(ego_points_coarse.shape[0])[:self.num_queries[0]]
                query_coord = ego_points_coarse[indices].reshape(B, -1, 3)
        
        for i, layer in enumerate(self.decoder_layers):
            if i != 0:
                if 'res_depths_mask' not in locals():
                    res_depths_mask = torch.ones_like(mask, dtype=torch.bool)
                
                mask = torch.logical_and(mask, torch.logical_and(roi_mask, res_depths_mask))

            if i == 1:
                if mask.sum().item() >= self.num_queries[1]:

                    ego_points_medium = ego_points[mask]
                    
                    # Check for nan values
                    if torch.isnan(ego_points_medium).any():
                        print("Warning: NaN detected in ego_points_medium")
                        ego_points_medium = torch.nan_to_num(ego_points_medium, nan=0.0)

                    try:
                        selected_ego_points_medium = farthest_point_sampling(
                            ego_points_medium, 
                            torch.tensor([ego_points_medium.shape[0]], device=ego_points_medium.device, dtype=torch.int),
                            torch.tensor([self.num_queries[1]], device=ego_points_medium.device, dtype=torch.int)
                        )
                        query_coord_medium = ego_points_medium[selected_ego_points_medium].reshape(B, -1, 3)
                    except Exception as e:
                        print(f"Warning: FPS failed for medium layer: {e}, using random sampling")
                        indices = torch.randperm(ego_points_medium.shape[0])[:self.num_queries[1]]
                        query_coord_medium = ego_points_medium[indices].reshape(B, -1, 3)
                else:

                    selected_ego_points_medium = selected_ego_points[self.num_queries[0]:self.num_queries[0]+self.num_queries[1]]
                    query_coord_medium = ego_points_coarse[selected_ego_points_medium].reshape(B, -1, 3)

            elif i == 2:
                try:
                    mask = torch.logical_and(mask, torch.logical_and(res_depths_mask, roi_mask))
                except Exception as e:
                    print(f"Warning: Mask combination failed at fine layer: {e}")
                    pass
                    
                if mask.sum().item() >= self.num_queries[2]:
                    ego_points_fine = ego_points[mask]
                    
                    selected_ego_points_fine = farthest_point_sampling(
                        ego_points_fine, 
                        torch.tensor([ego_points_fine.shape[0]], device=ego_points_fine.device, dtype=torch.int),
                        torch.tensor([self.num_queries[2]], device=ego_points_fine.device, dtype=torch.int)
                    )
                    query_coord_fine = ego_points_fine[selected_ego_points_fine].reshape(B, -1, 3)
                    
                else:

                    selected_ego_points_fine = selected_ego_points[self.num_queries[0]+self.num_queries[1]:self.num_queries[0]+self.num_queries[1]+self.num_queries[2]]
                    query_coord_fine = ego_points_coarse[selected_ego_points_fine].reshape(B, -1, 3)

            else:
                query_coord_medium = None
                query_coord_fine = None

            if not self.use_anisotropy_encoding:
                anisotropy_info = None
                query_bbox = point2bbox(query_coord[:, :self.num_queries[0]], box_size=1.6)
                query_bbox = encode_bbox(query_bbox, pc_range=self.pc_range)                                              # [B, N, 6]

                if i == 1:
                    query_bbox_medium = point2bbox(query_coord_medium, box_size=0.8)                                      # [B, N, 6]
                    query_bbox_medium = encode_bbox(query_bbox_medium, pc_range=self.pc_range)                            # [B, N, 6]
                    query_bbox = torch.cat([query_bbox, query_bbox_medium], dim=1)
                    query_coord = torch.cat([query_coord, query_coord_medium], dim=1)
                
                if i == 2:
                    query_coord = torch.cat([query_coord, query_coord_fine], dim=1)
                    query_coord_medium = query_coord[:, self.num_queries[0]:self.num_queries[0]+self.num_queries[1]]
                    query_bbox_medium = point2bbox(query_coord_medium, box_size=0.8)
                    query_bbox_medium = encode_bbox(query_bbox_medium, pc_range=self.pc_range)
                    query_bbox_fine = point2bbox(query_coord_fine, box_size=0.4)                                          # [B, N, 6]
                    query_bbox_fine = encode_bbox(query_bbox_fine, pc_range=self.pc_range)                                # [B, N, 6]
                    query_bbox = torch.cat([query_bbox, query_bbox_medium, query_bbox_fine], dim=1)
            else:
                query_bbox = point2bbox(query_coord[:, :self.num_queries[0]], box_size=1.6)                               # [B, N, 6]
                query_bbox = encode_bbox(query_bbox, pc_range=self.pc_range)                                              # [B, N, 6]
                if i == 1:
                    query_bbox_medium = point2bbox(query_coord_medium, box_size=0.8)                                      # [B, N, 6]
                    query_bbox_medium = encode_bbox(query_bbox_medium, pc_range=self.pc_range)                            # [B, N, 6]
                    query_bbox = torch.cat([query_bbox, query_bbox_medium], dim=1)
                    query_coord = torch.cat([query_coord, query_coord_medium], dim=1)                    
                
                if i == 2:
                    query_coord = torch.cat([query_coord, query_coord_fine], dim=1)
                    query_coord_medium = query_coord[:, self.num_queries[0]:self.num_queries[0]+self.num_queries[1]]
                    query_bbox_medium = point2bbox(query_coord_medium, box_size=0.8)
                    query_bbox_medium = encode_bbox(query_bbox_medium, pc_range=self.pc_range)
                    query_bbox_fine = point2bbox(query_coord_fine, box_size=0.4)                                          # [B, N, 6]
                    query_bbox_fine = encode_bbox(query_bbox_fine, pc_range=self.pc_range)                                # [B, N, 6]
                    query_bbox = torch.cat([query_bbox, query_bbox_medium, query_bbox_fine], dim=1)
            
                if i == 0:
                    anisotropy_info = None
                elif i == 1:
                    default_scale = torch.ones(B, self.num_queries[1], 3, device=gaussian['gau_scales'].device) * 0.8
                    default_rotation = torch.zeros(B, self.num_queries[1], 4, device=gaussian['gau_rots'].device)
                    default_rotation[..., 0] = 1.0
                    anisotropy_info = {'scale': torch.cat([gaussian['gau_scales'], default_scale], dim=1), 'rotation': torch.cat([gaussian['gau_rots'], default_rotation], dim=1)}
                elif i == 2:
                    default_scale = torch.ones(B, self.num_queries[2], 3, device=gaussian['gau_scales'].device) * 0.4
                    default_rotation = torch.zeros(B, self.num_queries[2], 4, device=gaussian['gau_rots'].device)
                    default_rotation[..., 0] = 1.0
                    anisotropy_info = {'scale': torch.cat([gaussian['gau_scales'], default_scale], dim=1), 'rotation': torch.cat([gaussian['gau_rots'], default_rotation], dim=1)}
            query_feat_part = query_feat[:, :query_bbox.size(1)]

            query_feat_part = layer(query_feat_part, query_bbox, mlvl_feats, anisotropy_info, img_metas)

            gau_pred = self.gau_pred_heads[i](query_feat_part)

            if i == 0:
                gaussian = self.query_2_gaussian(gau_pred, range=[0.0, 6.4])
                query_coord = gaussian['delta_xyz'] + query_coord

            elif i == 1:
                gaussian = self.query_2_gaussian(gau_pred[:self.num_queries[0]], range=[0.0, 6.4])
                query_coord[:self.num_queries[0]] = gaussian['delta_xyz'] + query_coord[:self.num_queries[0]]

                gaussian_medium = self.query_2_gaussian(gau_pred[self.num_queries[0]:self.num_queries[0]+self.num_queries[1]], range=[0.0, 6.4])
                query_coord[self.num_queries[0]:self.num_queries[0]+self.num_queries[1]] = gaussian_medium['delta_xyz'] / 2 + query_coord[self.num_queries[0]:self.num_queries[0]+self.num_queries[1]]

            elif i == 2:
                gaussian = self.query_2_gaussian(gau_pred[:self.num_queries[0]], range=[0.0, 6.4])
                query_coord[:self.num_queries[0]] = gaussian['delta_xyz'] + query_coord[:self.num_queries[0]]

                gaussian_medium = self.query_2_gaussian(gau_pred[self.num_queries[0]:self.num_queries[0]+self.num_queries[1]], range=[0.0, 6.4])
                query_coord[self.num_queries[0]:self.num_queries[0]+self.num_queries[1]] = gaussian_medium['delta_xyz'] / 2 + query_coord[self.num_queries[0]:self.num_queries[0]+self.num_queries[1]]

                gaussian_fine = self.query_2_gaussian(gau_pred[self.num_queries[0]+self.num_queries[1]:self.num_queries[0]+self.num_queries[1]+self.num_queries[2]], range=[0.0, 6.4])
                query_coord[self.num_queries[0]+self.num_queries[1]:self.num_queries[0]+self.num_queries[1]+self.num_queries[2]] = gaussian_fine['delta_xyz'] / 4 + query_coord[self.num_queries[0]+self.num_queries[1]:self.num_queries[0]+self.num_queries[1]+self.num_queries[2]]
            
            if self.render_conf['use_ov']:
                ov_query_feat = query_feat_part
                for ov_layer in self.ov_heads[i]:
                    ov_query_feat = ov_layer(ov_query_feat)
            else:
                ov_query_feat = None

            pred_gaussians = GaussianPrediction(
                means=query_coord,                   # [B, N, 3]
                scales=gaussian['gau_scales'],       # [B, N, 1]
                rotations=gaussian['gau_rots'],      # [B, N, 4]
                opacities=gaussian['gau_opacities'], # [B, N, 1]
                ovs=ov_query_feat,                          # [B, N, 512]
                colors=None,
            )

            gau_preds.append(pred_gaussians)

            with torch.no_grad():
                render_results = batch_splatting_render(pred_gaussians, W2C, render_k, render_conf=self.render_conf, inference=True)
            
                render_depths = render_results['depth']
                
                if torch.isnan(render_depths).any():
                    print("Warning: NaN detected in render_depths")
                    render_depths = torch.nan_to_num(render_depths, nan=0.0)
                                
                render_depths = render_depths.permute(0, 3, 1, 2)
    
                try:
                    depth_diff = render_depths - depth
                    if torch.isnan(depth_diff).any():
                        print("Warning: NaN detected in depth difference")
                        depth_diff = torch.nan_to_num(depth_diff, nan=0.0)
                    
                    res_depths_mask = depth_diff > 0.2
                    res_depths_mask = res_depths_mask.squeeze(1)
                    
                    if res_depths_mask.shape != mask.shape:
                        print(f"Warning: res_depths_mask shape mismatch: {res_depths_mask.shape} vs {mask.shape}")
                        res_depths_mask = torch.ones_like(mask, dtype=torch.bool)
                        
                except Exception as e:
                    print(f"Warning: Depth mask calculation failed: {e}")
                    res_depths_mask = torch.ones_like(mask, dtype=torch.bool)
            
        return gau_preds


class SparseGaussiansDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 pc_range=None,
                 self_attn=True,
                 past_queries=None,
                 use_anisotropy_encoding=True):
        super().__init__()

        self.use_anisotropy_encoding = use_anisotropy_encoding
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims), 
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        
        if self_attn:
            self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range, scale_adaptive=True, past_queries=past_queries)
            self.norm1 = nn.LayerNorm(embed_dims)
        else:
            self.self_attn = None
        
        self.sampling = SparseBEVSampling(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range,
            use_anisotropy_encoding=use_anisotropy_encoding
        )

        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=num_groups,
            out_points=num_points * num_frames * num_groups
        )

        self.ffn = FFN(embed_dims, feedforward_channels=embed_dims * 2, ffn_drop=0.1)
        
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        if self.self_attn is not None:
            self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()

    def forward(self, query_feat, query_3dgs, mlvl_feats, anisotropy_info, img_metas):
        query_pos = self.position_encoder(query_3dgs[..., :3])
        query_feat = query_feat + query_pos
        if self.self_attn is not None:
            query_feat = self.norm1(self.self_attn(query_3dgs, query_feat))
        sampled_feat = self.sampling(query_3dgs, query_feat, mlvl_feats, anisotropy_info, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))

        query_feat = self.norm3(self.ffn(query_feat))
        return query_feat
        
