# Generate info files manually
import os
import mmcv
import tqdm
import pickle
import argparse
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='data/nuscenes')
parser.add_argument('--version', default='v1.0-test')
args = parser.parse_args()


def get_cam_info(nusc, sample_data):
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    
    sensor2ego_translation = cs_record['translation']
    ego2global_translation = pose_record['translation']
    sensor2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    ego2global_rotation = Quaternion(pose_record['rotation']).rotation_matrix
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    sensor2global_rotation = sensor2ego_rotation.T @ ego2global_rotation.T
    sensor2global_translation = sensor2ego_translation @ ego2global_rotation.T + ego2global_translation

    return {
        'data_path': os.path.join(args.data_root, sample_data['filename']),
        'sensor2global_rotation': sensor2global_rotation,
        'sensor2global_translation': sensor2global_translation,
        'cam_intrinsic': cam_intrinsic,
        'timestamp': sample_data['timestamp'],
    }


def add_sweep_info(nusc, sample_infos):
    for curr_id in tqdm.tqdm(range(len(sample_infos['infos']))):
        sample = nusc.get('sample', sample_infos['infos'][curr_id]['token'])
        
        scene_token = sample['scene_token']
        scene_name = nusc.get('scene', scene_token)['name']
        sample_infos['infos'][curr_id]['scene_name'] = scene_name

    return sample_infos


if __name__ == '__main__':
    nusc = NuScenes(args.version, args.data_root)

    if args.version == 'v1.0-trainval':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_train.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_train.pkl'))

        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_val.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_val.pkl'))

    elif args.version == 'v1.0-test':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_test_sweep.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_test_sweep.pkl'))

    else:
        raise ValueError
