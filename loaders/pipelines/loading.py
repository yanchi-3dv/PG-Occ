import os
import mmcv
import glob
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from PIL import Image
from torchvision import transforms
from pyquaternion import Quaternion

class Voxelize():
    def __init__(self, max_volume_space, min_volume_space, grid_size, roi_mask=True):
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.grid_size = grid_size

        self.max_bound = np.asarray(self.max_volume_space)
        self.min_bound = np.asarray(self.min_volume_space)

        # get grid index
        self.crop_range = self.max_bound - self.min_bound

        self.intervals = self.crop_range / (self.grid_size)
        self.roi_mask = roi_mask

        if (self.intervals == 0).any():
            print("Zero interval!")

    def __call__(self, xyz, return_mask=False):

        if self.roi_mask:
            mask = np.all((xyz >= self.min_bound) & (xyz <= self.max_bound), axis=1)
            xyz = xyz[mask]

        grid_ind_float = (np.clip(xyz, self.min_bound, self.max_bound - 1e-3) - self.min_bound) / self.intervals
        grid_ind_float = grid_ind_float.astype(float)

        if return_mask:
            return grid_ind_float, mask
        return grid_ind_float

def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 ov_mode=False,
                 render_inf=False): 
        self.ov_mode = ov_mode
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        self.train_interval = [4, 8]
        self.test_interval = 6
        self.render_inf = render_inf

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        if self.render_inf:
            results['render_color_gt_orgin'] = [img.copy() for img in results['img'][0:6]]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    if 'ego2img' in results:
                        results['ego2img'].append(results['ego2img'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))

                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])
                        
            if self.render_inf:
                for j in range(len(cam_types)):
                    results['render_color_gt_orgin'].append(results['render_color_gt_orgin'][j])
                    results['cam2global'].append(results['cam2global'][j])
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            
            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])

            if self.render_inf:
                sweep_idx = 0
                sweep = results['sweeps']['prev'][sweep_idx]
                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['render_color_gt_orgin'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))    
                    results['cam2global'].append(sweep[sensor]['cam2global'])

        if self.render_inf:
            if len(results['sweeps']['next']) == 0:
                for j in range(len(cam_types)):
                    results['render_color_gt_orgin'].append(results['render_color_gt_orgin'][j])
                    results['cam2global'].append(results['cam2global'][j])
            else:
                sweep_idx = 0
                sweep = results['sweeps']['next'][sweep_idx]
                for sensor in cam_types:
                    results['render_color_gt_orgin'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))    
                    results['cam2global'].append(sweep[sensor]['cam2global'])
                                 
        return results

    def load_online(self, results):
        assert self.test_mode
        assert self.test_interval % 6 == 0

        if self.render_inf:
            results['render_color_gt_orgin'] = [img.copy() for img in results['img'][0:6]]

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    
                    if 'cam2global' in results:
                        results['cam2global'].append(np.copy(results['cam2global'][j]))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results
            
        world_size = get_dist_info()[1]
        if (world_size == 1 and self.test_mode):
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __init__(self, num_classes=18, inst_class_ids=[]):
        self.num_classes = num_classes
        self.inst_class_ids = inst_class_ids
    
    def __call__(self, results):
        occ_labels = np.load(results['occ_path'])
        semantics = occ_labels['semantics']  # [200, 200, 16]
        mask_camera = occ_labels['mask_camera'].astype(np.bool_)  # [200, 200, 16]

        results['mask_camera'] = mask_camera

        if 'instances' in occ_labels.keys():
            instances = occ_labels['instances']
            instance_class_ids = [self.num_classes - 1]
            for i in range(1, instances.max() + 1):
                class_id = np.unique(semantics[instances == i])
                assert class_id.shape[0] == 1, "each instance must belong to only one class"
                instance_class_ids.append(class_id[0])
            instance_class_ids = np.array(instance_class_ids)
        else:
            instances = None
            instance_class_ids = None

        instance_count = 0
        final_instance_class_ids = []
        final_instances = np.ones_like(semantics) * 255  # empty space has instance id "255"

        for class_id in range(self.num_classes - 1):
            if np.sum(semantics == class_id) == 0:
                continue

            if class_id in self.inst_class_ids:
                assert instances is not None, 'instance annotation not found'
                for instance_id in range(len(instance_class_ids)):
                    if instance_class_ids[instance_id] != class_id:
                        continue
                    final_instances[instances == instance_id] = instance_count
                    instance_count += 1
                    final_instance_class_ids.append(class_id)
            else:
                final_instances[semantics == class_id] = instance_count
                instance_count += 1
                final_instance_class_ids.append(class_id)

        results['voxel_semantics'] = semantics
        results['voxel_instances'] = final_instances
        results['instance_class_ids'] = DC(to_tensor(final_instance_class_ids))

        return results

@PIPELINES.register_module()
class LoadLidarFromFiles(object):
    def __init__(self, only_xyz=True, to_ego_coo=True, generate_gt=True, pc_range=[], occ_size=[]):
        self.only_xyz = only_xyz
        self.to_ego_coo = to_ego_coo
        self.generate_gt = generate_gt
        self.occ_size = occ_size
        self.fill_label = 0
        self.pc_range = pc_range
        if generate_gt:
            self.voxelization = Voxelize(pc_range[3:], pc_range[:3], occ_size)

@PIPELINES.register_module()
class LoadMultiViewSemanticFromFiles(object):
    def __init__(self):
        self.semantic_map = np.array([
            0,   # ignore
            4,   # sedan      -> car
            11,  # highway    -> driveable_surface
            3,   # bus        -> bus
            10,  # truck      -> truck
            14,  # terrain    -> terrain
            16,  # tree       -> vegetation
            13,  # sidewalk   -> sidewalk
            2,   # bicycle    -> bycycle
            1,   # barrier    -> barrier
            7,   # person     -> pedestrian
            15,  # building   -> manmade
            6,   # motorcycle -> motorcycle
            5,   # crane      -> construction_vehicle
            9,   # trailer    -> trailer
            8,   # cone       -> traffic_cone
            17   # sky        -> ignore
        ], dtype=np.int8)

    def __call__(self, results):
        semantics_2d = []

        for semantic_path in results['semantic_filename']:
            try:
                semantic = np.fromfile(semantic_path, dtype=np.int8).reshape(900, 1600)
                semantic = self.semantic_map[semantic]
                semantics_2d.append(semantic)
            except:
                print(semantic_path)

        results.update({'semantic_2d': semantics_2d})

        return results


@PIPELINES.register_module()
class GenerateRenderImageFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 render_conf=None,
                 test_mode=False):
        
        self.sweeps_num = sweeps_num    # 7
        assert render_conf is not None
        self.render_conf = render_conf
        self.test_mode = test_mode

        self.interp = transforms.InterpolationMode.LANCZOS
        self.resize = transforms.Resize((self.render_conf.render_h, self.render_conf.render_w), interpolation=self.interp)

    def load_render(self, results):

        if 'ori_k' in results.keys():
            results['ori_k'] = np.stack(results['ori_k'])   # (48, 3, 3)

        render_k = results['ori_k'].copy()
        render_k[..., 0, :] *= self.render_conf.render_w / results['ori_shape'][1]
        render_k[..., 1, :] *= self.render_conf.render_h / results['ori_shape'][0]
        
        render_k_4x4 = []

        for i in range(48):
            k_4x4 = np.eye(4)
            k_4x4[:3, :3] = render_k[i%6]
            render_k_4x4.append(k_4x4)

        results['render_k'] = render_k_4x4

        results['render_gt'] = []

        if 'depth' in results.keys():
            from scipy.ndimage import zoom
            original_depth = results['depth']
            zoom_factors = (1, self.render_conf['render_h']/900, self.render_conf['render_w']/1600)
            resized_depth = zoom(original_depth, zoom_factors, order=1)
            resized_depth = np.expand_dims(resized_depth, axis=1)
            results['depth'] = resized_depth

        if not self.test_mode:
            for img_array in results['render_color_gt_orgin']:
                img = Image.fromarray(img_array.astype('uint8'))
                results['render_gt'].append(np.array((self.resize(img))))

            results['auxi_img'] = []
            for auxi_img_path in results['img_auxi_paths']:
                auxi_img = mmcv.imread(auxi_img_path, 'color')
                auxi_img = Image.fromarray(auxi_img.astype('uint8'))
                results['auxi_img'].append(np.array((self.resize(auxi_img))))

            results['render_gt'] = np.array(results['render_gt'])

            results['cam2global'] = np.array(results['cam2global'])

            t0_2_global = results['cam2global'][0:6]
            tn_2_global = results['cam2global'][6:]

            t0_2_global_expanded = np.tile(t0_2_global, ((tn_2_global.shape[0] // t0_2_global.shape[0]), 1, 1))            
            t0_2_x_geo = np.linalg.inv(tn_2_global) @ t0_2_global_expanded
            
            results.update({"t0_2_x_geo": t0_2_x_geo})
        else:
            for img_array in results['img']:
                img = Image.fromarray(img_array.astype('uint8'))
                results['render_gt'].append(np.array((self.resize(img))))
            results['render_gt'] = np.array(results['render_gt'])

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return

        return self.load_render(results)


@PIPELINES.register_module()
class LoadFeatureFromFiles(object):
    def __init__(self, key='depth', root_path=None):
        self.key = key
        self.root_path = root_path

    def __call__(self, results):
        features = []
        for feature_name in results['feature_names']:
            if self.key == 'gt_depth':
                path = self.root_path + '/samples/' + feature_name.split("__")[1] + feature_name
            else:
                path = self.root_path + feature_name
            try:
                feature = np.load(path)
            except:
                print(f"key: {self.key}", f"feature_path {path} not found.")
            features.append(feature.astype(np.float32))

        features = np.array(features)
        results.update({self.key: features})
        return results

@PIPELINES.register_module()
class LoadSAMFromFiles(object):
    def __init__(self, key='sam', root_path=None):
        self.key = key
        self.root_path = root_path

    def __call__(self, results):
        feature_all = np.zeros((0, 900, 1600))
        cam_ids = np.zeros((0))
        for cam_id, feature_name in enumerate(results['feature_names']):
            path = self.root_path + feature_name
            try:
                feature = np.load(path)
                cam_ids = np.concatenate([cam_ids, np.ones((feature.shape[0])) * cam_id], axis=0)                    
                feature_all = np.concatenate([feature_all, feature], axis=0)
            except:
                pass
        assert cam_ids.shape[0] == feature_all.shape[0]
        results.update({self.key: feature_all, 'sam_cam_ids': cam_ids})
        return results

@PIPELINES.register_module()
class LoadOVFromFiles(object):
    def __init__(self, pc_range = [], occ_size = []):
        self.occ_size = occ_size
        self.voxelization = Voxelize(pc_range[3:], pc_range[:3], occ_size)
    
    def __call__(self, results):
        matching_points = []
        for matching_points_cam_path in results['matching_points_cam_filename']:
            matching_points_cam = np.load(matching_points_cam_path)
            matching_points.append(matching_points_cam)
        
        matching_points = np.concatenate(matching_points)
        if 'lidar_points_ego' in results.keys():
            ov_points_ego = results['lidar_points_ego'][matching_points]

        ov_features = []
        for ov_feature_path in results['ov_feature_filename']:
            ov_feature = np.load(ov_feature_path)
            ov_features.append(ov_feature)

        ov_features = np.vstack(ov_features)

        ov_grid_ind_float, mask = self.voxelization(ov_points_ego, return_mask=True)
        ov_grid_ind = np.floor(ov_grid_ind_float).astype(int)
        ov_features = ov_features[mask]

        ov_features = DC(to_tensor(ov_features))
        ov_grid_ind = DC(to_tensor(ov_grid_ind))
        
        results.update({'ov_features': ov_features,
                        'ov_grid_ind': ov_grid_ind})
        return results