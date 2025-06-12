#!/usr/bin/env python3
import numpy as np
import json
import copy
import os
from os import path
import sys

RAW_INTR = sys.argv[1]
RAW_POSE = sys.argv[2]

NORM_POSE = sys.argv[3]
NORM_CAMDICT = sys.argv[4]


def get_tf_cams(cam_dict, target_radius=1.):
    cam_centers = []
    for im_name in cam_dict:
        W2C = np.array(cam_dict[im_name]['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
        # tmp = -np.matmul(W2C[:3,:3].transpose(), W2C[:3,3:]).reshape(1,3)
        # print(cam_centers[0], tmp)  # they should be equal!
        # 1/0

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        #
        #
        #
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal

    translate = -center
    scale = target_radius / radius

    return translate, scale


def normalize_cam_dict(target_radius=1.):
    in_cam_dict = {}
    for camIdx, camName in enumerate(sorted(os.listdir(RAW_POSE))):
        print('processing', camName)
        tmp = np.loadtxt(path.join(RAW_POSE, camName))
        W2C = np.eye(4)
        if tmp.shape[0] == 3:
            W2C[:3,:4] = tmp
        else:
            W2C[:4,:4] = tmp
        # TODO: try inverting stuff, if it doesn't work
        print(W2C)
        viewName = path.splitext(camName)[0]+'.png'
        # viewName = f'{camIdx}.png'
        in_cam_dict[viewName] = {}
        in_cam_dict[viewName]['W2C'] = list(W2C.flatten())

        tmp = np.loadtxt(path.join(RAW_INTR, camName))
        intrinsics = tmp[:4, :4]
        in_cam_dict[viewName]['K'] = list(intrinsics.flatten())
        in_cam_dict[viewName]['img_size'] = [346, 260]
        in_cam_dict[viewName]['dist'] = list(tmp[4].flatten())

    translate, scale = get_tf_cams(in_cam_dict, target_radius=target_radius)

    def transform_pose(W2C, translate, scale):
        C2W = np.linalg.inv(W2C)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return np.linalg.inv(C2W)

    out_cam_dict = copy.deepcopy(in_cam_dict)
    for img_name in out_cam_dict:
        W2C = np.array(out_cam_dict[img_name]['W2C']).reshape((4, 4))
        W2C = transform_pose(W2C, translate, scale)
        assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
        out_cam_dict[img_name]['W2C'] = list(W2C.flatten())

        os.makedirs(NORM_POSE, exist_ok=True)
        # cam dict should contain inverse of the nerf pose file contents
        np.savetxt(path.join(NORM_POSE, path.splitext(img_name)[0]+'.txt'), np.linalg.inv(W2C))

        # outPath = 'poses_inv'
        # os.makedirs(outPath, exist_ok=True)
        # np.savetxt(path.join(outPath, f'{img_name.replace(".png", "")}.txt'), W2C)

    with open(NORM_CAMDICT, 'w') as fp:
        json.dump(out_cam_dict, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    normalize_cam_dict(target_radius=1/1.1)
    # cam dict should contain inverse of the nerf pose file contents
