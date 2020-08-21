import numpy as np
import time
import matplotlib.pyplot as plt

import torch

class TorchManipulator(object):
    def __init__(self, scale_sdev=0, rot_sdev=0, deflection=0, device='cuda',
                 mode='bilinear'):
        '''
        scale_sdev: Deviation of sampled scale factors (2 ** (0 +- sdev))
        rot_sdev: Deviation of sampled rotation in radians for 2D rotation
        deflection: 3D random rotation parameter between 0 (no rotation)
        and 1 (completely random rotation)
        '''

        if device == 'cuda' and not torch.cuda.is_available():
            raise Exception('CUDA device not available')
        self.device = device
        print('Torch transformer on device -> {}'.format(self.device))
        self.transformer = AffineTransformer(mode=mode)
        self.transformer.to(self.device)

        self.scale_sdev = scale_sdev
        self.rot_sdev = rot_sdev
        self.deflection = deflection

    def augment(self, data, labels=None):
        if data.ndim == 4:
            thetas = self.make_random_2D_transforms(dim=data.shape[0])
        elif data.ndim == 5:
            thetas = self.make_random_3D_transforms(dim=data.shape[0])
        else:
            raise Exception('Provide either 2D data (NxHxWxC) or 3D data (NxHxWxDxC).')

        data = self.transform(data, thetas)
        if labels is not None:
            labels = self.transform(labels, thetas)
            return data, labels
        else:
            return data

    def make_random_2D_transforms(self, dim):
        Ts = np.zeros((dim, 3, 3))
        for i in range(dim):
            S = 2 ** np.random.normal(loc=0, scale=self.scale_sdev)
            angle = np.random.normal(loc=0, scale=self.rot_sdev)
            Ts[i] = self.compose_2D_transform(S, angle)

        return Ts

    def compose_2D_transform(self, scale, rot):
        T = np.eye(3)
        R = np.array([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot),  0],
            [0,           0,            1]
            ])
        S = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0,     1]
            ])

        return np.matmul(S, np.matmul(R, T)).T

    def make_random_3D_transforms(self, dim):
        Ts = np.zeros((dim, 4, 4))
        for i in range(dim):
            S = 2 ** np.random.normal(loc=0, scale=self.scale_sdev)
            deflection = self.deflection
            Ts[i] = self.compose_3D_transform(S, deflection)

        return Ts

    def compose_3D_transform(self, scale, deflection):
        T = np.eye(4)
        R = self.get_random_3D_rotation(deflection)
        S = np.array([
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0],
            [0, 0, 0,     1]
            ])

        return np.matmul(S, np.matmul(R, T)).T[np.newaxis, :, :]

    def get_random_3D_rotation(self, deflection):
        # mostly taken from http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html

        theta, phi, z = np.random.uniform(size=(3,))
        theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0 * deflection  # For magnitude of pole deflection.
        r = np.sqrt(z)
        Vx, Vy, Vz = V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
            )
        st = np.sin(theta)
        ct = np.cos(theta)
        R = np.array(((-ct, -st, 0), (st, -ct, 0), (0, 0, 1))) # somehow, the R matrix is reflecting x and y dims when it should not
        M = np.eye(4)
        M[:3, :3] = (np.outer(V, V) - np.eye(3)).dot(R)
        return M

    def transform(self, data, theta):
        theta = np.linalg.inv(theta)

        if data.ndim == 4 and theta.shape == (data.shape[0], 3, 3):
            theta = torch.tensor(theta[:, :2, :], dtype=torch.float).to(self.device)
            tensor = torch.from_numpy(data).to(self.device).float().permute(0,3,1,2)

            transformed_tensor = self.transformer(tensor, theta)
            transformed_tensor = transformed_tensor.permute(0,2,3,1).cpu().numpy()

        elif data.ndim == 5 and theta.shape == (data.shape[0], 4, 4):
            theta = torch.tensor(theta[:, :3, :], dtype=torch.float).to(self.device)
            tensor = torch.from_numpy(data).to(self.device).float().permute(0,4,1,2,3)

            transformed_tensor = self.transformer(tensor, theta)
            transformed_tensor = transformed_tensor.permute(0,2,3,4,1).cpu().numpy()

        else:
            raise Exception('Incompatible shapes. Provide either 2D data (NxHxWxC) with theta (Nx3x3) or 3D data (NxHxWxDxC) with theta (Nx4x4)')

        return transformed_tensor

class AffineTransformer(torch.nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, tensor, theta):
        grid = torch.nn.functional.affine_grid(theta,
            size=tensor.shape, align_corners=False)
        tensor = torch.nn.functional.grid_sample(
            tensor, grid, mode=self.mode, padding_mode="zeros", align_corners=False)

        return tensor
