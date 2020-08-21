import numpy as np
import os
import argparse
import datetime
import pickle
import sys

import torch
import nrrd
import scipy.ndimage as sn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from landmark_detection_training import torch_nets, torch_transform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-paths', help='File containing lines with paths to target .nrrd binary skull volumes.')
    parser.add_argument('--implant-path', default=None, help='If path to implant folder is specified, applies the same transformation to corresponding implant volumes.')
    parser.add_argument('--output-path', help='Output path to save the aligned .nrrd files.')
    parser.add_argument('--model-path', help='Path to landmark detection cnn weights.')
    parser.add_argument('--save-previews', action='store_true', help='Save previews of the alignment results.')
    # default values should be left alone when using up-to-date model, although they could be experimented with in combination with different landmark detection model training
    parser.add_argument('--voxel-size', type=float, default=0.4, help='Target voxel size.')
    parser.add_argument('-n', '--net', default='UNet_3D_4_16', help='See net_definitions for options.')
    parser.add_argument('--gauss-sigma', type=float, default=3, help='Sigma for gaussians in landmark ground truth maps.')
    parser.add_argument('--patch-size', type=int, default=96, help="Shape of training data patches.")
    args = parser.parse_args()
    return args


def point_detection(vol, sigma):
    correlations = sn.gaussian_filter(vol, sigma=sigma)
    pred = np.argmax(correlations)
    pred = np.unravel_index(pred, correlations.shape)
    return pred


def print_timestamp():
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    print('\n\n[{}-{}-{} {}:{}]\n'.format(year, month, day, hour, minute))


def apply_transform(data, T, target_shape=(512, 512, 512)):
    transformed_data = np.zeros(target_shape)
    sn.interpolation.affine_transform(
        data.astype(np.float),
        T,
        output_shape=target_shape,
        order=0,
        output=transformed_data
        )
    return (transformed_data > 0.5).astype(np.float)


def get_rigid_transform(lm_moving, shape=(512, 512, 512), z_offset=0, return_static_landmarks=False):
    '''
    Get rotation and translation of detected landmarks to static landmarks
    which lay on z=0 plane of the volume using SVD (part of ICP algorithm).
    '''
    # define target landmark positions to align the target skull to
    lm_static = np.array(((3*shape[0]//4, shape[1]//2, z_offset),
                          (shape[0]//4, shape[1]//2, z_offset),
                          (3*shape[0]//5, shape[1]//5, z_offset),
                          (2*shape[0]//5, shape[1]//5, z_offset)))

    centroid_s = np.mean(lm_static, axis=0)
    centroid_m = np.mean(lm_moving, axis=0)
    lm_static_cen = lm_static - centroid_s
    lm_moving_cen = lm_moving - centroid_m

    H = np.dot(lm_moving_cen.T, lm_static_cen)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:  # special reflection case
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    R_hat = np.identity(4)
    R_hat[:3, :3] = R

    T_hat = np.identity(4)
    T_hat[:3, 3] = centroid_s - np.dot(R, centroid_m)

    M = np.dot(T_hat, R_hat)

    if return_static_landmarks:
        return M, lm_static
    else:
        return M


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

    if not os.path.exists(os.path.join(args.output_path, 'aligned_skull')):
        os.makedirs(os.path.join(args.output_path, 'aligned_skull'))
    if not os.path.exists(os.path.join(args.output_path, 'aligned_implant')) and args.implant_path is not None:
        os.makedirs(os.path.join(args.output_path, 'aligned_implant'))
    if args.save_previews:
        preview_path = os.path.join(args.output_path, 'previews')
        if not os.path.exists(preview_path):
            os.makedirs(preview_path)

    with open(args.case_paths) as handle:
        case_paths = handle.read().splitlines()

    print('Torch init')
    manipulator = torch_transform.TorchManipulator()

    model = torch_nets.LandmarkNet(
        args.net,
        output_channels=4,
        input_channels=1
        )
    model.set_eval()
    model.load_weights(args.model_path)
    print('Loaded model {}.\n'.format(args.model_path))

    for case_path in case_paths:
        case_id = os.path.splitext(os.path.split(case_path)[1])[0]
        print('Aligning case {}'.format(case_path))

        data, header = nrrd.read(case_path)
        S = get_scale_transform(header, target_voxel_spacing=args.voxel_size)

        pred = model.infer_full_volume(data, pad=args.patch_size)
        pred_lms = np.zeros((4, 3))
        for i in range(4):
            pred_lms[i, :] = point_detection(
                pred[:, :, :, i], sigma=args.gauss_sigma)
        R = get_rigid_transform(np.dot(pred_lms, S[:3, :3]), data.shape)

        T = np.dot(R, S)
        transformed_data = apply_transform(
            data, np.linalg.inv(T), target_shape=(512, 512, 512))

        nrrd.write(
            os.path.join(args.output_path, 'aligned_skull', '{}.nrrd'.format(case_id)), transformed_data)
        print('Saved aligned skull volume.')

        if args.implant_path is not None:
            implant, _ = nrrd.read(
                os.path.join(args.implant_path, '{}.nrrd'.format(case_id)))
            transformed_implant = apply_transform(
                implant, np.linalg.inv(T), target_shape=(512, 512, 512))
            nrrd.write(
                os.path.join(args.output_path, 'aligned_implant', '{}.nrrd'.format(case_id)), transformed_implant)
            print('Saved aligned implant volume.')

        if args.save_previews:
            plt.subplot(221)
            plt.imshow(np.average(data, axis=1), cmap='Greys')
            plt.subplot(222)
            plt.imshow(np.average(transformed_data, axis=1), cmap='Greys')
            plt.subplot(223)
            plt.imshow(np.average(data, axis=2), cmap='Greys')
            plt.subplot(224)
            plt.imshow(np.average(transformed_data, axis=2), cmap='Greys')
            plt.savefig(
                os.path.join(preview_path, '{}.jpg'.format(case_id)), dpi=300)
            plt.close()


if __name__ == '__main__':
    main()
