import os
import pickle
import time
import subprocess
import re
import sys
import numpy as np
import argparse
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import scipy.ndimage as sn
import nrrd
from scipy.spatial import distance

import landmark_dataset
import torch_nets
import torch_transform

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tst-data', required=True, help='Path to test volume and annotation path pairs file.')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint to restore.')
    parser.add_argument('--output-path', required=True, help='Path to output.')

    parser.add_argument('--patch-size', type=int, default=96, help="Shape of training data patches.")
    parser.add_argument('-g', '--gpu-id', type=int, help="If not set setGPU() is called. Set this to 0 on desktop. Leave it empty on SGE.")
    parser.add_argument('-n', '--net', default='UNet_3D_4_16', help='See net_definitions for options.')
    parser.add_argument('--gauss-sigma', type=float, default=3, help='Sigma for gaussians in landmark ground truth maps.')
    parser.add_argument('--scale-sdev', type=float, default=0.1, help='Scale factor deviation for data augmentation')
    parser.add_argument('--deflection', type=float, default=0.2, help='3D random rotation parameter between 0 (no rotation) and 1 (completely random rotation)')

    args = parser.parse_args()
    return args

def print_timestamp():
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    print('\n\n[{}-{}-{} {}:{}]\n'.format(year, month, day, hour, minute))

def setGPU():
    freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    freeGpu = freeGpu.decode().strip()
    if len(freeGpu) == 0:
        print('no free GPU!')
        sys.exit(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu
    print('got GPU ' + (freeGpu))

    return(int(freeGpu.strip()))

def point_detection(vol, sigma):
    correlations = sn.gaussian_filter(vol, sigma=sigma)
    pred = np.argmax(correlations)
    pred = np.unravel_index(pred, correlations.shape)
    return pred

def get_scale_transform(header, target_voxel_spacing=0.4):
    '''
    Get transformation matrix that rescales the volume to target isometric
    resolution.
    '''
    spacing = header['space directions']
    S = np.zeros((4, 4))
    S[0, 0] = np.linalg.norm(spacing[0]) / target_voxel_spacing
    S[1, 1] = np.linalg.norm(spacing[1]) / target_voxel_spacing
    S[2, 2] = np.linalg.norm(spacing[2]) / target_voxel_spacing
    S[3, 3] = 1
    return S

def main():
    print_timestamp()
    args = parse_arguments()

    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(os.path.join(args.output_path))

    with open(args.tst_data, 'r') as handle:
        test_cases = handle.read().splitlines()
    print('Loading train dataset')

    print('Torch init')
    model = torch_nets.LandmarkNet(
        args.net,
        output_channels=4,
        input_channels=1
        )
    model.set_train()
    model_path = args.checkpoint
    model.load_weights(model_path)
    print('Loaded model {}.\n'.format(model_path))

    if args.gpu_id is None:
        setGPU()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    model.set_eval()

    tst_lam_dists = []
    tst_ram_dists = []
    tst_lsn_dists = []
    tst_rsn_dists = []
    # get test metrics
    for case in test_cases:
        case_path, annotation_path = case.split('\t')
        data, header = nrrd.read(case_path)
        with open(annotation_path, 'rb') as f:
            annotations = pickle.load(f)
        S = get_scale_transform(header, target_voxel_spacing=1)

        pred = model.infer_full_volume(data, pad=args.patch_size)
        pred[pred<0] = 0

        point_pred = point_detection(pred[:,:,:,0], sigma=args.gauss_sigma)
        lam_dist = distance.euclidean(
            np.dot(annotations['left_auditory_meatus'], S[:3, :3]),
            np.dot(point_pred, S[:3, :3])
            )
        tst_lam_dists.append(lam_dist)
        point_pred = point_detection(pred[:,:,:,1], sigma=args.gauss_sigma)
        ram_dist = distance.euclidean(
            np.dot(annotations['right_auditory_meatus'], S[:3, :3]),
            np.dot(point_pred, S[:3, :3])
            )
        tst_ram_dists.append(ram_dist)
        point_pred = point_detection(pred[:,:,:,2], sigma=args.gauss_sigma)
        lsn_dist = distance.euclidean(
            np.dot(annotations['left_supraorbital_notch'], S[:3, :3]),
            np.dot(point_pred, S[:3, :3])
            )
        tst_lsn_dists.append(lsn_dist)
        point_pred = point_detection(pred[:,:,:,3], sigma=args.gauss_sigma)
        rsn_dist = distance.euclidean(
            np.dot(annotations['right_supraorbital_notch'], S[:3, :3]),
            np.dot(point_pred, S[:3, :3])
            )
        tst_rsn_dists.append(rsn_dist)

    errors = {
        'Left AM error': tst_lam_dists,
        'Right AM error': tst_ram_dists,
        'Left SON error': tst_lsn_dists,
        'Right SON error': tst_rsn_dists,
        }
    with open(os.path.join(args.output_path, 'logs.pkl'), 'wb') as handle:
        pickle.dump(errors, handle)

    fig, ax = plt.subplots()
    ax.boxplot(errors.values())
    ax.set_xticklabels(errors.keys())
    # plt.ylim(0, 5)
    plt.savefig(os.path.join(args.output_path, 'log_eval.jpg'), dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
