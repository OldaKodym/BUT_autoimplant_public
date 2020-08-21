import numpy as np
import os
import argparse
import pickle

import matplotlib.pyplot as plt
import nrrd

def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--data-path', required=True, help='Path to AutoImplant challenge data.')

    return args.parse_args()

def save_preview(data, path, case):
    if not os.path.exists(os.path.join(path, 'previews')):
        os.makedirs(os.path.join(path, 'previews'))

    plt.imshow(np.average(data, axis=1), cmap='Greys')
    plt.savefig(os.path.join(path, 'previews', '{}.png'.format(case)), dpi=300)

class AnnotationTool(object):
    def __init__(self, data, annotations=None):

        self.fig, ax_array = plt.subplots(1, 4, gridspec_kw={'wspace':0.1, 'hspace':0.1})
        self.ax_axi = ax_array[0]
        self.ax_sag = ax_array[1]
        self.ax_fro = ax_array[2]
        self.ax_proj = ax_array[3]

        self.frontal_projection = np.average(data, axis=1)

        if annotations is None:
            self.annotations = {
                'nasion': [],
                'left_auditory_meatus': [],
                'right_auditory_meatus': [],
                'sella_turcica': [],
                'left_supraorbital_notch': [],
                'right_supraorbital_notch': []
                }
        else:
            self.annotations = annotations

        self.point = []

        self.X = data.astype(np.float)
        self.rows, self.cols, self.slices = self.X.shape
        self.ind_sag = self.rows//2
        self.ind_fro = self.cols//2
        self.ind_axi = self.slices//2

        self.im_sag = self.ax_sag.imshow(self.X[self.ind_sag,:,:], cmap='gray', interpolation='nearest', vmin=0, vmax=1.0)
        self.im_fro = self.ax_fro.imshow(self.X[:,self.ind_fro,:], cmap='gray', interpolation='nearest', vmin=0, vmax=1.0)
        self.im_axi = self.ax_axi.imshow(self.X[:,:,self.ind_axi], cmap='gray', interpolation='nearest', vmin=0, vmax=1.0)
        self.im_proj = self.ax_proj.imshow(self.frontal_projection, cmap='gray', interpolation='nearest', vmin=0, vmax=1.0)

        self.update()

    def onkey(self, event):
        self.update()
        if event.key == 'd' and self.point:
            self.annotations['left_auditory_meatus'] = self.point
            print('Set current point ({}) to LEFT AUDITORY MEATUS landmark.'.format(self.point))

        elif event.key == 'g' and self.point:
            self.annotations['right_auditory_meatus'] = self.point
            print('Set current point ({}) to RIGHT AUDITORY MEATUS landmark.'.format(self.point))

        elif event.key == 'r' and self.point:
            self.annotations['nasion'] = self.point
            print('Set current point ({}) to NASION landmark.'.format(self.point))

        elif event.key == 'f' and self.point:
            self.annotations['sella_turcica'] = self.point
            print('Set current point ({}) to SELLA TURCICA landmark.'.format(self.point))

        elif event.key == 'e' and self.point:
            self.annotations['left_supraorbital_notch'] = self.point
            print('Set current point ({}) to LEFT SUPRAORBITAL NOTCH landmark.'.format(self.point))

        elif event.key == 't' and self.point:
            self.annotations['right_supraorbital_notch'] = self.point
            print('Set current point ({}) to RIGHT SUPRAORBITAL NOTCH landmark.'.format(self.point))

        elif event.key == 'x' and self.point:
            if (self.annotations['left_auditory_meatus']
                and self.annotations['right_auditory_meatus']
                and self.annotations['nasion']
                and self.annotations['sella_turcica']
                and self.annotations['left_supraorbital_notch']
                and self.annotations['right_supraorbital_notch']):

                plt.close()
            else:
                print('Annotation not complete.')
                print(self.annotations)
        self.update()

    def onscroll(self, event):
        if event.inaxes == self.ax_axi:
            if event.button == 'up':
                self.ind_axi = (self.ind_axi + 1) % self.slices
            else:
                self.ind_axi = (self.ind_axi - 1) % self.slices

        elif event.inaxes == self.ax_sag:
            if event.button == 'up':
                self.ind_sag = (self.ind_sag + 1) % self.rows
            else:
                self.ind_sag = (self.ind_sag - 1) % self.rows

        elif event.inaxes == self.ax_fro:
            if event.button == 'up':
                self.ind_fro = (self.ind_fro + 1) % self.cols
            else:
                self.ind_fro = (self.ind_fro - 1) % self.cols

        self.update()

    def onmove(self, event):
        if event.inaxes == self.ax_axi and event.button != 3:
            self.ind_sag = int(event.ydata)
            self.ind_fro = int(event.xdata)

        elif event.inaxes == self.ax_sag and event.button != 3:
            self.ind_axi = int(event.xdata)
            self.ind_fro = int(event.ydata)

        elif event.inaxes == self.ax_fro and event.button != 3:
            self.ind_axi = int(event.xdata)
            self.ind_sag = int(event.ydata)

        elif event.inaxes == self.ax_proj and event.button != 3:
            self.ind_axi = int(event.xdata)
            self.ind_sag = int(event.ydata)

        self.update()

    def onclick(self, event):
        if event.button == 1:
            print(self.point)
            self.point = [self.ind_sag, self.ind_fro, self.ind_axi]
        self.update()

    def update(self):
        img_sag = self.X[self.ind_sag, :, :].copy()
        img_sag[self.ind_fro, :] = 0.5
        img_sag[:, self.ind_axi] = 0.5
        img_fro = self.X[:, self.ind_fro, :].copy()
        img_fro[self.ind_sag, :] = 0.5
        img_fro[:, self.ind_axi] = 0.5
        img_axi = self.X[:, :, self.ind_axi].copy()
        img_axi[self.ind_sag, :] = 0.5
        img_axi[:, self.ind_fro] = 0.5
        img_proj = self.frontal_projection.copy()
        img_proj[self.ind_sag, :] = 0.5
        img_proj[:,self.ind_axi] = 0.5
        self.im_sag.set_data(img_sag)
        self.im_fro.set_data(img_fro)
        self.im_axi.set_data(img_axi)
        self.im_proj.set_data(img_proj)

        self.im_sag.axes.figure.canvas.draw()

    def show(self):
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show()

def main():
    args = parse_arguments()

    lm_dir = os.path.join(args.data_path, 'landmark_annotations')
    if not os.path.exists(lm_dir):
        os.makedirs(lm_dir)

    for case in os.listdir(args.data_path):
        print(case)
        if not os.path.exists(os.path.join(lm_dir, '{}.pkl'.format(case))):
            data, meta = nrrd.read(os.path.join(args.data_path, case))
            # save_preview(data, args.data_path, case)
            annotator = AnnotationTool(data)
            annotator.show()
        else:
            data, meta = nrrd.read(os.path.join(args.data_path, case))
            # save_preview(data, args.data_path, case)
            with open(os.path.join(lm_dir, '{}.pkl'.format(case)), 'rb') as handle:
                annotations = pickle.load(handle)
            annotator = AnnotationTool(data, annotations=annotations)
            annotator.show()

        if (annotator.annotations['left_auditory_meatus']
            and annotator.annotations['right_auditory_meatus']
            and annotator.annotations['nasion']
            and annotator.annotations['sella_turcica']
            and annotator.annotations['left_supraorbital_notch']
            and annotator.annotations['right_supraorbital_notch']):

            with open(os.path.join(lm_dir, '{}.pkl'.format(case)), 'wb') as handle:
                pickle.dump(annotator.annotations, handle)
            print('Annotations for case {} saved.'.format(case))
        else:
            print('Annotations for case {} not complete, not saved.'.format(case))


if __name__ == '__main__':
    main()
