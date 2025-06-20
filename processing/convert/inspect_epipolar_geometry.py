#!/usr/bin/env python3
import cv2
import numpy as np
import os
import json
import imageio as io


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def two_view_geometry(intrinsics1, extrinsics1, intrinsics2, extrinsics2):
    relative_pose = extrinsics2.dot(np.linalg.inv(extrinsics1))
    R = relative_pose[:3, :3]
    T = relative_pose[:3, 3]
    tx = skew(T)
    E = np.dot(tx, R)
    F = np.linalg.inv(intrinsics2[:3, :3]).T.dot(E).dot(np.linalg.inv(intrinsics1[:3, :3]))

    return E, F, relative_pose


def drawpointslines(img1, pts1, img2, lines2, colors):
    h, w = img2.shape[:2]
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    print(pts1.shape, lines2.shape, colors.shape)
    for p, l, c in zip(pts1, lines2, colors):
        c = tuple(c.tolist())
        img1 = cv2.circle(img1, tuple(p), 5, c, -1)

        x0, y0 = map(int, [0, -l[2]/l[1]])
        x1, y1 = map(int, [w, -(l[2]+l[0]*w)/l[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), c, 1, lineType=cv2.LINE_AA)
    return img1, img2


def inspect(img1, K1, W2C1, img2, K2, W2C2):
    E, F, relative_pose = two_view_geometry(K1, W2C1, K2, W2C2)

    orb = cv2.ORB_create()
    kp1 = orb.detect(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)[:5]
    pts1 = np.array([[int(kp.pt[0]), int(kp.pt[1])] for kp in kp1])

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    colors = np.random.randint(0, high=255, size=(len(pts1), 3))

    img1, img2 = drawpointslines(img1, pts1, img2, lines2, colors)

    im_to_show = np.concatenate((img1, img2), axis=1)
    # down sample to fit screen
    h, w = im_to_show.shape[:2]
    scale = 3
    im_to_show = cv2.resize(im_to_show, (int(scale*w), int(scale*h)), interpolation=cv2.INTER_AREA)
    cv2.imshow('epipolar geometry', im_to_show)
    while cv2.waitKey(0) != ord('q'):
        pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import os
    import sys
    base_dir = sys.argv[1]
    # base_dir = '.'

    img_dir = os.path.join(base_dir, 'rgb')
    cam_dict_file = os.path.join(base_dir, 'camdict.json')
    # img_name2 = '0.png'
    # img_name1 = 'r_00500.png'
    # img_name1 = '1.png'

    cam_dict = json.load(open(cam_dict_file))

    all_cameras = list(sorted(cam_dict.keys()))
    all_images = list(sorted(os.listdir(img_dir)))

    idx1 = int(sys.argv[2])
    idx2 = int(sys.argv[3])

    img_name1 = all_images[idx1]
    img_name2 = all_images[idx2]

    cam_name1 = all_cameras[idx1]
    cam_name2 = all_cameras[idx2]

    # img1 = cv2.imread(os.path.join(img_dir, img_name1))
    img1 = io.imread(os.path.join(img_dir, img_name1))
    if len(img1.shape)>3:
        img1[..., :3] *= img1[..., 3:]>0
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    # K1 = np.array(cam_dict[img_name1]['K']).reshape((5, 4))[:4,:4]
    K1 = np.array(cam_dict[cam_name1]['K']).reshape((4, 4))[:4,:4]
    dist1 = np.array(cam_dict[cam_name1]['dist'])
    W2C1 = np.array(cam_dict[cam_name1]['W2C']).reshape((4, 4))
    img1 = cv2.undistort(img1, K1[:3,:3], dist1)

    # img2 = cv2.imread(os.path.join(img_dir, img_name2))
    img2 = io.imread(os.path.join(img_dir, img_name2))
    if len(img2.shape)>3:
        img2[..., :3] *= img2[..., 3:]>0
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    # K2 = np.array(cam_dict[img_name2]['K']).reshape((5, 4))[:4,:4]
    K2 = np.array(cam_dict[cam_name2]['K']).reshape((4, 4))[:4,:4]
    dist2 = np.array(cam_dict[cam_name2]['dist'])
    W2C2 = np.array(cam_dict[cam_name2]['W2C']).reshape((4, 4))
    img2 = cv2.undistort(img2, K2[:3,:3], dist2)

    inspect(img1, K1, W2C1, img2, K2, W2C2)

