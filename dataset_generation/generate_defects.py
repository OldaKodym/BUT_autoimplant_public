import numpy as np
import os
import argparse
import datetime

import nrrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as sn

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-paths', help='File containing lines with paths to target .nrrd binary skull volumes.')
    parser.add_argument('--output-path', help='Output path to save the .nrrd volumes with defects.')
    parser.add_argument('--num-defects', type=int, default=5, help='How many defects to generate for each case.')
    parser.add_argument('--save-preview', action='store_true', help='Also save defective skulls previews to the output path.')

    parser.add_argument('--offset', type=int, default=80, help='Only place defects on this z-coordinate and farther to focus the defects to cranial area.')
    parser.add_argument('--r1-mean', type=float, default=70, help='Mean radius of primary sphere.')
    parser.add_argument('--r1-sdev', type=float, default=20, help='STD of primary sphere radius.')
    parser.add_argument('--r2-mean', type=float, default=40, help='Mean radius of secondary spheres.')
    parser.add_argument('--r2-sdev', type=float, default=10, help='STD of secondary spheres radius.')
    parser.add_argument('--min-secondary', type=int, default=0, help='Minimum number of secondary spheres.')
    parser.add_argument('--max-secondary', type=int, default=8, help='Maximum number of secondary spheres.')

    args = parser.parse_args()
    return args

class DefectGenerator(object):
    '''
    This class generates random combinations of spheres that serve as synthetic skull defect shapes.
    '''
    def __init__(self, r1_mean, r1_sdev, r2_mean, r2_sdev, min_sec, max_sec,
                 elastic_alpha=300, elastic_sigma=10):
        self.r1_mean = r1_mean
        self.r1_sdev = r1_sdev
        self.r2_mean = r2_mean
        self.r2_sdev = r2_sdev
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.alpha = elastic_alpha
        self.sigma = elastic_sigma

        self.pregenerated_distmap = np.ones((512,512,512))
        self.pregenerated_distmap[256,256,256] = 0
        self.pregenerated_distmap = sn.morphology.distance_transform_edt(
            self.pregenerated_distmap)

    def elastic_transform(self, image, random_state=None):
        assert len(image.shape)==2

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape

        dx = self.alpha * sn.filters.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            self.sigma, mode="constant", cval=0
            )
        dy = self.alpha * sn.filters.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            self.sigma, mode="constant", cval=0
            )

        x, y = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

        return sn.interpolation.map_coordinates(
            image, indices, order=1).reshape(shape)

    def generate_defect(self, size):
        distmap_crop = self.pregenerated_distmap[:,:,256-size[2]//2:256-size[2]//2+size[2]]

        # this makes a sphere with random radius for primary defect
        volume_1 = np.zeros(size)
        volume_1[distmap_crop<np.random.normal(self.r1_mean, self.r1_sdev)] = 1
        volume_1 = volume_1.astype(np.bool)

        # these are surface positions on which secondary defects are added
        volume_surface = volume_1 ^ sn.morphology.binary_erosion(volume_1)
        surface_inds = np.where(volume_surface)

        # add random number of secondary defect shapes
        volume_2 = np.ones_like(volume_1)
        for _ in range(np.random.randint(self.min_sec, self.max_sec)):
            ind = np.random.randint(len(surface_inds[0]-1))
            volume_2[surface_inds[0][ind], surface_inds[1][ind], surface_inds[2][ind]] = 0
        volume_2 = sn.morphology.distance_transform_edt(volume_2) < np.random.normal(self.r2_mean, self.r2_sdev)

        # final defect shape is morphologically open combination of primary and secondary shapes
        volume = volume_1 | volume_2
        volume = sn.morphology.binary_opening(volume, iterations=5)

        # apply random elastic deformation in two planes
        state = np.random.randint(512)
        for i in range(volume.shape[0]):
            if np.amax(volume[i,:,:]) > 0:
                volume[i,:,:] = self.elastic_transform(
                    volume[i,:,:].astype(np.float),
                    random_state=np.random.RandomState(state)
                    )
        state = np.random.randint(512)
        for i in range(volume.shape[1]):
            if np.amax(volume[:,i,:]) > 0:
                volume[:,i,:] = self.elastic_transform(
                    volume[:,i,:].astype(np.float),
                    random_state=np.random.RandomState(state)
                    )

        return volume > 0.5

def main():
    args = parse_arguments()

    defective_skull_path = os.path.join(args.output_path, 'defective_skull')
    implant_path = os.path.join(args.output_path, 'implant')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(defective_skull_path):
        os.makedirs(defective_skull_path)
    if not os.path.exists(implant_path):
        os.makedirs(implant_path)

    with open(args.case_paths, 'r') as handle:
        cases_to_break = handle.read().splitlines()

    defect_generator = DefectGenerator(
        r1_mean=args.r1_mean,
        r1_sdev=args.r1_sdev,
        r2_mean=args.r2_mean,
        r2_sdev=args.r2_sdev,
        min_sec=args.min_secondary,
        max_sec=args.max_secondary
    )

    for case in cases_to_break:
        data, _ = nrrd.read(case)
        case_id = os.path.splitext(os.path.split(case)[1])[0]

        x_coords, y_coords, z_coords = np.where(data>0)
        x_coords = [x_coord for x_coord, z_coord in zip(x_coords, z_coords) if z_coord >= args.offset]
        y_coords = [y_coord for y_coord, z_coord in zip(y_coords, z_coords) if z_coord >= args.offset]
        z_coords = [z_coord for z_coord in z_coords if z_coord >= args.offset]

        for i in range(args.num_defects):
            defect = defect_generator.generate_defect(size=data.shape)

            coord_ind = np.random.randint(len(x_coords))
            point = [x_coords[coord_ind] + np.random.randint(-32, 32),
                     y_coords[coord_ind] + np.random.randint(-32, 32),
                     z_coords[coord_ind] + np.random.randint(-32, 32)]
            defect_shifted = sn.interpolation.shift(
                defect, (point[0]-256, point[1]-256, point[2]-data.shape[2]//2), order=0)

            data_defective_skull = data.copy()
            data_defective_skull[defect_shifted] = 0

            data_implant = data.copy()
            data_implant[~defect_shifted] = 0

            nrrd.write(
                os.path.join(defective_skull_path, '{}_{}.nrrd'.format(case_id, i)),
                data_defective_skull
                )
            nrrd.write(
                os.path.join(implant_path, '{}_{}.nrrd'.format(case_id, i)),
                data_implant
                )

            if args.save_preview:
                plt.subplot(221)
                plt.imshow(np.average(data, axis=1))
                plt.subplot(222)
                plt.imshow(np.average(defect, axis=1))
                plt.subplot(223)
                plt.imshow(np.average(data_defective_skull, axis=1))
                plt.subplot(224)
                plt.imshow(np.average(data_implant, axis=1))
                plt.savefig(os.path.join(args.output_path, '{}_{}.jpg'.format(case_id, i)))

            print('Saved defective skull and implant ({}) for case {}.'.format(i, case_id))

if __name__ == '__main__':
    main()
