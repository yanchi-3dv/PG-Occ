import os
import re
import numpy as np
import torch
import mmcv
from tqdm import tqdm
from torch.utils.data import DataLoader
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from .nuscenes_occ_dataset import NuSceneOcc
from .ego_pose_dataset import EgoPoseDataset
from .ray_metrics import main_rayiou, main_raypq
from .old_metrics import main_miou
from configs.pgocc import occ_class_names as occ3d_class_names

@DATASETS.register_module()
class NuSceneOVOcc(NuSceneOcc):
    """
    NuScenes Open-Vocabulary Occupancy Dataset.
    
    Extends NuSceneOcc with open-vocabulary features and temporal information.
    """
    def __init__(self, render_conf, return_intrinsic, metric, next_frame=1, *args, **kwargs):
        """
        Initialize NuSceneOVOcc dataset.
        
        Args:
            render_conf: Rendering configuration dictionary
            return_intrinsic: Whether to return camera intrinsics
            metric: List of metrics to evaluate
            next_frame: Number of future frames to include
        """
        super().__init__(*args, **kwargs)

        self.return_intrinsic = return_intrinsic
        self.metric = metric
        self.render_conf = render_conf
        self.next_frame = next_frame

        # Camera types in nuScenes
        self.cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

    def collect_sweeps(self, index, into_past=150, into_future=1):
        """
        Collect temporal sweeps (past and future frames).
        
        Args:
            index: Current frame index
            into_past: Maximum number of past frames to collect
            into_future: Maximum number of future frames to collect
        
        Returns:
            all_sweeps_prev: List of past sweeps
            all_sweeps_next: List of future sweeps
        """
        # Collect past sweeps
        all_sweeps_prev = []
        curr_index = index
        scene_name = self.data_infos[curr_index]['scene_name']
        
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index -= 1
        
        # Collect future sweeps
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            # Ensure we stay in the same scene
            future_scene_name = self.data_infos[curr_index]['scene_name']
            if future_scene_name != scene_name:
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index += 1
        
        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        """
        Get data info for a specific frame.
        
        Args:
            index: Frame index
        
        Returns:
            input_dict: Dictionary containing all frame data
        """
        info = self.data_infos[index]
        
        # Collect temporal sweeps
        sweeps_prev, sweeps_next = self.collect_sweeps(index, into_future=self.next_frame)
        
        # Extract pose information
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        # Initialize input dictionary
        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )
        
        # Compute ego to lidar transformation
        ego2lidar = transform_matrix(
            lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)
        input_dict['ego2lidar'] = [ego2lidar for _ in range(6)]
        
        # Set occupancy ground truth path
        input_dict['occ_path'] = os.path.join(
            self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')

        # Process camera data
        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            ego2img_rts = []
            cam2ego_rts = []
            feature_names = []
            ori_ks = []
            img_auxi_paths = []
            cam2global_rts = []

            for _, cam_info in info['cams'].items():
                # Basic image information
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_auxi_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # Camera to ego transformation
                cam2ego = np.eye(4)
                cam2ego[:3, :3] = cam_info['sensor2ego_rotation']
                cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                cam2ego_rts.append(cam2ego)
                cam2global_rts.append(cam_info['cam2global'])

                # Feature names for preprocessed data
                feature_names.append(re.sub(
                    r'data/nuscenes/samples/CAM_\w+', "", 
                    cam_info['data_path'][:-4] + '.npy'))

                # Lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                # Ego to image transformation
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                
                ego2cam = np.linalg.inv(cam2ego)
                ego2img = viewpad @ ego2cam
                ego2img_rts.append(ego2img)

                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

                if self.return_intrinsic:
                    ori_ks.append(intrinsic)

            # Update input dictionary with camera data
            input_dict.update(dict(
                img_filename=img_paths,
                feature_names=feature_names,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                ego2img=ego2img_rts,
                cam2ego=cam2ego_rts,
                cam2global=cam2global_rts, 
                img_auxi_paths=img_auxi_paths,
            ))
        
            if self.return_intrinsic:
                input_dict.update(dict(ori_k=ori_ks))

        # Process lidar data
        if self.modality['use_lidar']:
            lidar_path = info['lidar_path'].replace("/code/project/SparseOcc/", "./")
            input_dict.update(dict(lidar_path=lidar_path))

        # Process temporal auxiliary frames (for training only)
        if not self.test_mode:
            past_frame_num = self.render_conf['ov_auxi_past_frame_num']
            future_frame_num = self.render_conf['ov_auxi_future_frame_num']
            
            if past_frame_num > 0 or future_frame_num > 0:
                current_scene_name = info['scene_name']
                
                # Collect past frames
                if past_frame_num > 0:
                    for past_frame_idx in range(past_frame_num):
                        frame_idx = index - past_frame_idx - 1
                        if frame_idx < 0:
                            break
                        
                        past_info = self.data_infos[frame_idx]
                        past_scene_name = past_info['scene_name']
                        
                        # Ensure we stay in the same scene
                        if past_scene_name != current_scene_name:
                            break

                        # Process each camera in the past frame
                        for cam_idx, (_, cam_past_info) in enumerate(past_info['cams'].items()):
                            cam_past2global = cam_past_info['cam2global']
                            # Transform past camera to current ego coordinate
                            cam_past2ego = (cam2ego_rts[cam_idx] @ 
                                          np.linalg.inv(cam2global_rts[cam_idx]) @ 
                                          cam_past2global)
                            cam2ego_rts.append(cam_past2ego)
                            img_auxi_paths.append(os.path.relpath(cam_past_info['data_path']))
                        
                        input_dict.update(dict(
                            cam2ego=cam2ego_rts,
                            img_auxi_paths=img_auxi_paths,
                        ))

        # Add scene name
        input_dict['scene_name'] = info['scene_name']
        
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        """
        Evaluate occupancy prediction results.
        
        Args:
            occ_results: List of prediction results
            runner: Training runner (optional)
            show_dir: Directory to save visualizations (optional)
            **eval_kwargs: Additional evaluation arguments
        
        Returns:
            results: Dictionary of evaluation metrics
        """
        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins, cam_masks = [], [], [], [], [], []
        print('\nStarting Evaluation...')
        results = {}

        # Evaluate depth metrics if requested
        if 'depth' in self.metric:
            error = {cam_type: [] for cam_type in self.cam_types}
            for i, occ_result in enumerate(occ_results):
                for cam_idx, error_value in enumerate(occ_result['depth_error']):
                    error[self.cam_types[cam_idx]].append(error_value)

            # Print depth error metrics
            header = ("{:>8} | " * 7).format(
                "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
            print(header)
            
            mean_errors = []
            for cam in self.cam_types:
                error_data = np.array(error[cam])
                mean_error = np.nanmean(error_data, axis=0)
                mean_errors.append(mean_error)
                print(f"{cam:<15}: " + ("{:>8.3f} | " * 7).format(*mean_error))
                results[f'{cam}'] = mean_error

            # Compute average errors across all cameras
            total_errors = np.stack(mean_errors)
            average_total_errors = np.nanmean(total_errors, axis=0)
            print(f"{'average':<15}: " + ("{:>8.3f} | " * 7).format(*average_total_errors))
            results['average'] = average_total_errors
        
        sample_tokens = [info['token'] for info in self.data_infos]
        # Collect predictions and ground truth for occupancy evaluation
        for batch in DataLoader(EgoPoseDataset(self.data_infos), num_workers=8):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            # Load ground truth
            occ_path = os.path.join(
                self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']
            gt_mask_camera = occ_gt['mask_camera'].astype(bool)

            # Get prediction
            occ_pred = occ_results[data_id]
            
            # Determine class names based on dataset type
            data_type = self.occ_gt_root.split('/')[-1]
            if data_type in ['gts', 'occ3d_panoptic']:
                occ_class_names = occ3d_class_names
            else:
                raise ValueError(
                    f"Unsupported dataset type: {data_type}. "
                    f"Supported types: 'gts', 'occ3d_panoptic'")

            # Extract semantic predictions
            sem_pred = occ_pred['occ_preds'][0]
            try:
                sem_pred = sem_pred.squeeze(0).cpu().numpy()
            except:
                sem_pred = sem_pred

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
            cam_masks.append(gt_mask_camera)

        # Compute occupancy metrics
        if len(inst_preds) > 0:
            # Panoptic metrics
            results = main_raypq(
                occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins, 
                occ_class_names=occ_class_names)
            results.update(main_rayiou(
                occ_preds, occ_gts, lidar_origins, 
                occ_class_names=occ_class_names))
        else:
            # Semantic metrics only
            if 'rayiou' in self.metric:
                results.update(main_rayiou(
                    occ_preds, occ_gts, lidar_origins, 
                    occ_class_names=occ_class_names))
            if 'miou' in self.metric:
                results.update(main_miou(
                    occ_preds, occ_gts, cam_masks, 
                    occ_class_names=occ_class_names))
        
        return results

    def format_results(self, occ_results, submission_prefix, **kwargs):
        """
        Format results for submission.
        
        Args:
            occ_results: List of prediction results
            submission_prefix: Directory to save formatted results
            **kwargs: Additional arguments
        """
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, f'{sample_token}.npz')
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        
        print('\nFinished.')
