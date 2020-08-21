import os
import pickle
import numpy as np

import tables
import nrrd
import scipy.ndimage as nd

class LandmarkDataset(object):
    '''
    Re-saves the data and rendered landmark labels to h5 files in tmp path. Keeps
    the paths and landmark positions to make on-the-fly batch sampling efficient.
    '''
    def __init__(self, case_paths, tmp_path, sigma=2):
        self.case_paths = case_paths
        self.tmp_path = tmp_path
        self.sigma = sigma
        self.filt = tables.Filters(complib='zlib', complevel=1)

        self.cases = []
        self.data_paths = {}
        self.label_paths = {}
        self.annotations = {}

        self.prepare_tmp_data()

    def prepare_tmp_data(self):
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        for case in self.case_paths:
            case_path, annotation_path = case.split('\t')
            case_id = os.path.splitext(os.path.split(case_path)[1])[0]
            self.cases.append(case_id)

            data_path = os.path.join(self.tmp_path, '{}.h5'.format(case_id))
            label_path = os.path.join(self.tmp_path, '{}_lm.h5'.format(case_id))
            self.data_paths[case_id] = data_path
            self.label_paths[case_id] = label_path
            with open(annotation_path, 'rb') as f:
                annotations = pickle.load(f)
            case_annotations = {
                'left_auditory_meatus': annotations['left_auditory_meatus'],
                'right_auditory_meatus': annotations['right_auditory_meatus'],
                'left_supraorbital_notch': annotations['left_supraorbital_notch'],
                'right_supraorbital_notch': annotations['right_supraorbital_notch'],
            }
            self.annotations[case_id] = case_annotations

            if not (os.path.exists(data_path) and os.path.exists(label_path)):
                data, _ = nrrd.read(case_path)
                data_h5 = tables.open_file(
                    data_path,
                    mode='w',
                    filters=self.filt
                    )
                contents = data_h5.create_carray(
                    data_h5.root, 'chunks',
                    tables.Atom.from_dtype(data.dtype),
                    shape=data.shape,
                    chunkshape=[8,8,8]
                    )
                contents[:] = data
                data_h5.close()

                labels = self.render_skull_landmarks(data.shape, annotations)
                labels_h5 = tables.open_file(
                    label_path,
                    mode='w',
                    filters=self.filt
                    )
                contents = labels_h5.create_carray(
                    labels_h5.root, 'chunks',
                    tables.Atom.from_dtype(labels.dtype),
                    shape=labels.shape,
                    chunkshape=[8,8,8,1]
                    )
                contents[:] = labels
                labels_h5.close()

            print('Saved case {} in {}.'.format(case_id, self.tmp_path))

    def render_skull_landmarks(self, shape, annotations):
        canvas = np.zeros((shape[0], shape[1], shape[2], 4), dtype=np.float)
        canvas[annotations['left_auditory_meatus'][0],
               annotations['left_auditory_meatus'][1],
               annotations['left_auditory_meatus'][2],
               0] = 1
        canvas[annotations['right_auditory_meatus'][0],
               annotations['right_auditory_meatus'][1],
               annotations['right_auditory_meatus'][2],
               1] = 1
        canvas[annotations['left_supraorbital_notch'][0],
               annotations['left_supraorbital_notch'][1],
               annotations['left_supraorbital_notch'][2],
               2] = 1
        canvas[annotations['right_supraorbital_notch'][0],
               annotations['right_supraorbital_notch'][1],
               annotations['right_supraorbital_notch'][2],
               3] = 1
        for i in range(4):
            canvas[:, :, :, i] = nd.gaussian_filter(
                canvas[:, :, :, i], sigma=self.sigma)
            canvas[:, :, :, i] /= np.amax(canvas[:, :, :, i])
        return canvas

    def get_batch(self, batch_size, batch_shape, seed=None):
        shp = batch_shape
        random_state = np.random.RandomState(seed=seed)

        batch_data = np.zeros((batch_size, shp[0], shp[1], shp[2], 1))
        batch_labels = np.zeros((batch_size, shp[0], shp[1], shp[2], 4))

        for i in range(batch_size):
            # pick a case
            case = random_state.choice(self.cases)

            data_h5 = tables.open_file(
                os.path.join(self.tmp_path, '{}.h5'.format(case)),
                mode='r+',
                filters=self.filt
                )
            labels_h5 = tables.open_file(
                os.path.join(self.tmp_path, '{}_lm.h5'.format(case)),
                mode='r+',
                filters=self.filt
                )

            # pick a center point from:
            vol_shp = data_h5.root.chunks.shape
            random_num = random_state.rand()

            if random_num > 0.8: # anywhere
                ind_x = random_state.randint(shp[0]//2, vol_shp[0]-shp[0]//2+1)
                ind_y = random_state.randint(shp[1]//2, vol_shp[1]-shp[1]//2+1)
                ind_z = random_state.randint(shp[2]//2, vol_shp[2]-shp[2]//2+1)
            else: # landmark points
                lm = random_state.choice(list(self.annotations[case].keys()))
                point = self.annotations[case][lm]
                ind_x = point[0] - random_state.randint(-shp[0]//2+1, shp[0]//2)
                ind_x = np.clip(ind_x, shp[0]//2, vol_shp[0]-shp[0]//2)
                ind_y = point[1] - random_state.randint(-shp[1]//2+1, shp[1]//2)
                ind_y = np.clip(ind_y, shp[1]//2, vol_shp[1]-shp[1]//2)
                ind_z = point[2] - random_state.randint(-shp[2]//2+1, shp[2]//2)
                ind_z = np.clip(ind_z, shp[2]//2, vol_shp[2]-shp[2]//2)

            # fill the batch
            batch_data[i, :, :, :, 0] = data_h5.root.chunks[
                ind_x-shp[0]//2: ind_x+shp[0]//2,
                ind_y-shp[0]//2: ind_y+shp[0]//2,
                ind_z-shp[0]//2: ind_z+shp[0]//2
            ]
            batch_labels[i, :, :, :, :] = labels_h5.root.chunks[
                ind_x-shp[0]//2: ind_x+shp[0]//2,
                ind_y-shp[0]//2: ind_y+shp[0]//2,
                ind_z-shp[0]//2: ind_z+shp[0]//2,
                :
            ]

            data_h5.close()
            labels_h5.close()

            # maybe flip laterally
            if random_state.rand() > 0.5:
                batch_data[i, :, :, :, 0] = np.flip(
                    batch_data[i, :, :, :, 0], axis=0)
                batch_labels[i, :, :, :, :] = np.flip(
                    batch_labels[i, :, :, :, :], axis=0)
                # switch left/right landmarks
                batch_labels[i, :, :, :, 0:2] = batch_labels[i, :, :, :, 1::-1]
                batch_labels[i, :, :, :, 2:4] = batch_labels[i, :, :, :, 3:1:-1]

        return batch_data, batch_labels

    def get_case(self, case):
        data_h5 = tables.open_file(
            os.path.join(self.tmp_path, '{}.h5'.format(case)),
            mode='r+',
            filters=self.filt
        )
        data = data_h5.root.chunks[:,:,:].copy()
        data_h5.close()

        return data, self.annotations[case]
