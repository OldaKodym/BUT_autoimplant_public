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
from scipy.spatial import distance

import landmark_dataset
import torch_nets
import torch_transform

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn-data', required=True, help='Path to training volume and annotation path pairs file.')
    parser.add_argument('--tst-data', required=True, help='Path to test volume and annotation path pairs file.')
    parser.add_argument('--tmp-path', required=True, help='Path to temporary data path utilized during training.')
    parser.add_argument('-o', '--output-path', required=True, help='Output path')
    parser.add_argument('--batch-size', type=int, help="Batch size.")
    parser.add_argument('--patch-size', type=int, help="Shape of training data patches.")

    parser.add_argument('--start-iteration', default=0, type=int)
    parser.add_argument('--max-iterations', default=200000, type=int)
    parser.add_argument('--view-step', default=5000, type=int)
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint to restore')
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

def main():
    print_timestamp()
    args = parse_arguments()

    if not os.path.exists(os.path.join(args.output_path, 'train_log')):
        os.makedirs(os.path.join(args.output_path, 'train_log'))
    if not os.path.exists(os.path.join(args.output_path, 'test_log')):
        os.makedirs(os.path.join(args.output_path, 'test_log'))
    if not os.path.exists(os.path.join(args.output_path, 'model')):
        os.makedirs(os.path.join(args.output_path, 'model'))

    with open(args.trn_data, 'r') as handle:
        train_cases = handle.read().splitlines()
    with open(args.tst_data, 'r') as handle:
        test_cases = handle.read().splitlines()
    print('Loading train dataset')
    train_dataset = landmark_dataset.LandmarkDataset(
        train_cases,
        tmp_path=args.tmp_path,
        sigma=args.gauss_sigma
        )
    print('Loading test dataset')
    test_dataset = landmark_dataset.LandmarkDataset(
        test_cases,
        tmp_path=args.tmp_path,
        sigma=args.gauss_sigma
        )

    print('Torch init')
    model = torch_nets.LandmarkNet(
        args.net,
        output_channels=4,
        input_channels=1
        )
    model.set_train()
    model_path = None
    if args.checkpoint:
        model_path = args.checkpoint
    else:
        if args.start_iteration > 1:
            model_path = os.path.join(args.output_path, 'model', 'checkpoint_{0:06d}.torch'.format(args.start_iteration))
    if model_path is not None:
        model.load_weights(model_path)
        print('Loaded model {}.\n'.format(model_path))
    else:
        print('No checkpoint loaded, training from scratch.\n')

    manipulator = torch_transform.TorchManipulator(
        scale_sdev=args.scale_sdev,
        deflection=args.deflection
        )

    if args.gpu_id is None:
        setGPU()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    trn_losses = []
    tst_losses = []

    print('\nBeginning training')
    for iteration in range(args.start_iteration, args.max_iterations + 1):
        data, labels = train_dataset.get_batch(
            args.batch_size,
            (args.patch_size, args.patch_size, args.patch_size)
            )
        data, labels = manipulator.augment(data, labels)
        pred, l = model.train_step(data, labels)

        if iteration % 100 == 0:
            model.set_eval()

            # get losses
            data, labels = train_dataset.get_batch(
                args.batch_size,
                (args.patch_size, args.patch_size, args.patch_size),
                seed=0
                )
            _, l_trn = model.train_step(data, labels, update=False)
            trn_losses.append(l_trn)
            data, labels = test_dataset.get_batch(
                args.batch_size,
                (args.patch_size, args.patch_size, args.patch_size),
                seed=0
                )
            _, l_tst = model.train_step(data, labels, update=False)
            tst_losses.append(l_tst)
            print('Iteration: {}, train loss: {}, test loss: {}'.format(iteration, l_trn, l_tst))

            model.set_train()

        if (iteration % args.view_step == 0 and iteration > args.start_iteration) or iteration == args.max_iterations:
            model.set_eval()

            checkpoint_path = os.path.join(args.output_path, 'model', 'checkpoint_{0:06d}.torch'.format(iteration))
            model.save_weights(checkpoint_path)
            print('\n\nModel weights saved to {}'.format(checkpoint_path))

            tst_lam_dists = []
            tst_ram_dists = []
            tst_lsn_dists = []
            tst_rsn_dists = []
            trn_lam_dists = []
            trn_ram_dists = []
            trn_lsn_dists = []
            trn_rsn_dists = []
            # get test metrics
            for case in test_dataset.cases:
                data, annotations = test_dataset.get_case(case)
                pred = model.infer_full_volume(data, pad=args.patch_size)
                pred[pred<0] = 0

                plt.subplot(231)
                plt.imshow(np.average(data, axis=1), cmap='Greys')

                plt.subplot(232)
                plt.imshow(np.average(data, axis=1), cmap='Greys')

                plt.subplot(233)
                plt.imshow(np.average(data, axis=1), cmap='Greys')

                plt.subplot(234)
                pred_0 = np.average(pred[:,:,:,0], axis=1)
                plt.imshow(pred_0)

                plt.subplot(235)
                pred_1 = np.average(pred[:,:,:,1]+pred[:,:,:,2], axis=1)
                plt.imshow(pred_1)

                plt.subplot(236)
                pred_2 = np.average(pred[:,:,:,3], axis=1)
                plt.imshow(pred_2)

                plt.savefig(os.path.join(args.output_path, 'test_log', '{}_out.jpg'.format(case)), dpi=300)

                point_pred = point_detection(pred[:,:,:,0], sigma=args.gauss_sigma)
                lam_dist = distance.euclidean(annotations['left_auditory_meatus'], point_pred)
                tst_lam_dists.append(lam_dist)
                point_pred = point_detection(pred[:,:,:,1], sigma=args.gauss_sigma)
                ram_dist = distance.euclidean(annotations['right_auditory_meatus'], point_pred)
                tst_ram_dists.append(ram_dist)
                point_pred = point_detection(pred[:,:,:,2], sigma=args.gauss_sigma)
                lsn_dist = distance.euclidean(annotations['left_supraorbital_notch'], point_pred)
                tst_lsn_dists.append(lsn_dist)
                point_pred = point_detection(pred[:,:,:,3], sigma=args.gauss_sigma)
                rsn_dist = distance.euclidean(annotations['right_supraorbital_notch'], point_pred)
                tst_rsn_dists.append(rsn_dist)

            errors = {
                'Left AM error': tst_lam_dists,
                'Right AM error': tst_ram_dists,
                'Left SON error': tst_lsn_dists,
                'Right SON error': tst_rsn_dists,
                }
            fig, ax = plt.subplots()
            ax.boxplot(errors.values())
            ax.set_xticklabels(errors.keys())
            # plt.ylim(0, 5)
            plt.savefig(os.path.join(args.output_path, 'test_log', 'log_{}.jpg'.format(iteration)), dpi=300)
            plt.close()
            # get train metrics
            for case in train_dataset.cases[:10]:
                data, annotations = train_dataset.get_case(case)
                pred = model.infer_full_volume(data, pad=args.patch_size)

                point_pred = point_detection(pred[:,:,:,0], sigma=args.gauss_sigma)
                lam_dist = distance.euclidean(annotations['left_auditory_meatus'], point_pred)
                trn_lam_dists.append(lam_dist)
                point_pred = point_detection(pred[:,:,:,1], sigma=args.gauss_sigma)
                ram_dist = distance.euclidean(annotations['right_auditory_meatus'], point_pred)
                trn_ram_dists.append(ram_dist)
                point_pred = point_detection(pred[:,:,:,2], sigma=args.gauss_sigma)
                lsn_dist = distance.euclidean(annotations['left_supraorbital_notch'], point_pred)
                trn_lsn_dists.append(lsn_dist)
                point_pred = point_detection(pred[:,:,:,3], sigma=args.gauss_sigma)
                rsn_dist = distance.euclidean(annotations['right_supraorbital_notch'], point_pred)
                trn_rsn_dists.append(rsn_dist)

            errors = {
                'Left AM error': trn_lam_dists,
                'Right AM error': trn_ram_dists,
                'Left SON error': trn_lsn_dists,
                'Right SON error': trn_rsn_dists,
                }
            fig, ax = plt.subplots()
            ax.boxplot(errors.values())
            ax.set_xticklabels(errors.keys())
            # plt.ylim(0, 5)
            plt.savefig(os.path.join(args.output_path, 'train_log', 'log_{}.jpg'.format(iteration)), dpi=300)
            plt.close()

            model.set_train()

if __name__ == '__main__':
    main()
