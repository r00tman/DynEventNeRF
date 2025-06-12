import os
import glob

import numpy as np
from PIL import Image

from nerf_sample_ray_split import RaySamplerSingleEventStream
from ddp_config import logger


def find_files(dir_, exts):
    if os.path.isdir(dir_):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir_, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def find_files_per_frames(dir_, exts, tstart=None, tend=None):
    '''Detect files in multi-frame dir structure.
    Returns list of lists of pairs of frame number and file path per frame per view

    Example: If dir_ is as follows
    - 0000
      - 0.png
      - 1.png
      - 2.png
    - 0100
      - 0.png
      - 1.png
      - 2.png
    Then the function returns
    [[(0, '0000/0.png'), (100, '0100/0.png')],
     [(0, '0000/1.png'), (100, '0100/1.png')],
     [(0, '0000/2.png'), (100, '0100/2.png')]]'''

    assert os.path.isdir(dir_)
    views_per_frame = []
    for frame in os.listdir(dir_):
        frame_number = int(frame)
        if tstart is not None and frame_number < tstart:
            continue
        if tend is not None and frame_number > tend:
            continue
        views = find_files(os.path.join(dir_, frame), exts)
        views_per_frame.append((frame_number, views))
    views_per_frame.sort()

    view_count = 0
    if views_per_frame:
        view_count = len(views_per_frame[0][1])
        # might not be necessary irl
        # but if not true, we need view matching logic
        # which we don't want to have yet
        assert all(len(views) == view_count for _, views in views_per_frame), 'number of views should match for every frame'

    frames_per_view = [[(frame_number, paths[view_idx]) for frame_number, paths in views_per_frame]
                                                        for view_idx in range(view_count)]

    return frames_per_view


def load_event_data_split(basedir, scene, split, camera_mgr, skip=1, view_filter=None,
                          use_ray_jitter=True,
                          polarity_offset=0.0,
                          damping_strength=0.93,
                          tstart=0., tend=1000.,
                          is_rgb_only=False):

    def parse_txt(filename, shape):
        assert os.path.isfile(filename)
        with open(filename) as f:
            nums = f.read().split()
        return np.array([float(x) for x in nums]).reshape(shape).astype(np.float32)

    if basedir[-1] == '/':          # remove trailing '/'
        basedir = basedir[:-1]

    split_dir = '{}/{}/{}'.format(basedir, scene, split)

    print(split_dir)
    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))

    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    cam_cnt = len(pose_files)

    # event file
    event_files = find_files('{}/events'.format(split_dir), exts=['*.npz'])
    event_files = event_files[::skip]
    print(event_files)
    assert len(event_files) == cam_cnt, f'len(event_files)={len(event_files)} != cam_cnt={cam_cnt}'

    # img files
    img_files = find_files_per_frames('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg', '*.JPG', '*.PNG'],
                                      tstart=tstart, tend=tend)
    if len(img_files) > 0:
        logger.info('raw img_files: {}'.format(len(img_files)))
        logger.info('number of frames: {}'.format(len(img_files[0])))
        img_files = img_files[::skip]
        assert len(img_files) == cam_cnt
    else:
        img_files = [None, ] * cam_cnt

    # mask files
    mask_files = find_files('{}/mask'.format(split_dir), exts=['*.png', '*.jpg', '*.JPG', '*.PNG'])
    if len(mask_files) > 0:
        logger.info('raw mask_files: {}'.format(len(mask_files)))
        mask_files = mask_files[::skip]
        assert len(mask_files) == cam_cnt
    else:
        mask_files = [None, ] * cam_cnt

    # background files
    background_files = find_files('{}/background'.format(split_dir), exts=['*.png', '*.jpg', '*.JPG', '*.PNG'])
    if len(background_files) > 0:
        logger.info('raw background_files: {}'.format(len(background_files)))
        background_files = background_files[::skip]
        assert len(background_files) == cam_cnt
    else:
        background_files = [None, ] * cam_cnt

    # apply filter for the render view
    if view_filter:
        logger.info(f'Only loading view named "{view_filter}"')
        # 1. first find it in the rgb names
        idx = None
        for i, img_frames in enumerate(img_files):
            _, fn = img_frames[0]  # take the first frame
            if view_filter in fn:
                idx = i
                break
        assert idx is not None
        logger.info(f'Found "{view_filter}" in "{img_files[idx]}"')
        intrinsics_files = [intrinsics_files[idx]]
        pose_files = [pose_files[idx]]
        img_files = [img_files[idx]]
        mask_files = [mask_files[idx]]
        background_files = [background_files[idx]]
        event_files = [event_files[idx]]
        cam_cnt = 1

    # ----

    for i in range(cam_cnt):
        curr_file = img_files[i][0][1]  # take the first frame path
        if not camera_mgr.contains(curr_file):
            pose = parse_txt(pose_files[i], (4,4))
            camera_mgr.add_camera(curr_file, pose)


    # create ray samplers
    ray_samplers = []
    
    H, W = None, None
    for i in range(cam_cnt):
        try:
            intrinsics = parse_txt(intrinsics_files[i], (5,5))
        except ValueError:
            try:
                intrinsics = parse_txt(intrinsics_files[i], (5,4))
            except ValueError:
                intrinsics = parse_txt(intrinsics_files[i], (4,4))
                # no built-in distortion:
                # concat unity distortion coefficients
                intrinsics = np.concatenate((intrinsics, np.zeros((1,4), dtype=np.float32)), 0)

        event_data = np.load(event_files[i])
        xs, ys, ts, ps = event_data['x'], event_data['y'], event_data['t'], event_data['p']
        events = (xs, ys, ts, ps)

        # H, W = 260, 346

        curr_file = img_files[i]
        curr_mask = mask_files[i]
        curr_background = background_files[i]

        if H is None:
            # parse the height and width from the first image
            img = np.array(Image.open(curr_file[0][1]))
            H, W = img.shape[:2]
            logger.info(f'Detected camera resolution: {W}x{H}')


        ray_samplers.append(RaySamplerSingleEventStream(H=H, W=W, intrinsics=intrinsics,
                                                        events=events,
                                                        rgb_paths=curr_file,
                                                        mask_path=curr_mask,
                                                        background_path=curr_background,
                                                        use_ray_jitter=use_ray_jitter,
                                                        polarity_offset=polarity_offset,
                                                        damping_strength=damping_strength,
                                                        tstart=tstart, tend=tend,
                                                        is_rgb_only=is_rgb_only))

    logger.info('Split {}, # views: {}, # effective views: {}'.format(split, cam_cnt, len(ray_samplers)))

    return ray_samplers
