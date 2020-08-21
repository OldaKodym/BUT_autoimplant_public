from functools import partial
import numpy as np

import torch

class LandmarkNet(object):
    def __init__(self, net, output_channels=4, input_channels=1, device='cuda'):

        if device == 'cuda' and not torch.cuda.is_available():
            raise Exception('CUDA device not available')
        self.device = device
        print('Building net {} on device -> {}'.format(net, self.device))

        self.output_channels = output_channels
        self.net = net_definitions[net](outputs=output_channels, inputs=input_channels)
        self.net = self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss = torch.nn.MSELoss().to(self.device)

    def set_train(self):
        self.net = self.net.train()

    def set_eval(self):
        self.net = self.net.eval()

    def train_step(self, inputs, labels, update=True):
        inputs = torch.from_numpy(inputs).to(self.device).float().permute(0,4,1,2,3)
        labels = torch.from_numpy(labels).to(self.device).float().permute(0,4,1,2,3)
        if update:
            self.optimizer.zero_grad()
            out = self.net(inputs)
            loss = self.loss(out, labels)
            loss.backward()
            self.optimizer.step()
        else:
            out = self.net(inputs)
            loss = self.loss(out, labels)
        return out, loss.item()

    def save_weights(self, path):
        torch.save(self.net.state_dict(), path)

    def load_weights(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def pad_to_atleast(self, volume, size):
        pads = np.zeros((3), np.int)
        if volume.shape[0] < size:
            half_diff = int(np.ceil((size - volume.shape[0]) / 2))
            volume = np.pad(volume, ((half_diff,half_diff),(0,0),(0,0)), 'constant')
            pads[0] = half_diff
        if volume.shape[1] < size:
            half_diff = int(np.ceil((size - volume.shape[1]) / 2))
            volume = np.pad(volume, ((0,0),(half_diff,half_diff),(0,0)), 'constant')
            pads[1] = half_diff
        if volume.shape[2] < size:
            half_diff = int(np.ceil((size - volume.shape[2]) / 2))
            volume = np.pad(volume, ((0,0),(0,0),(half_diff,half_diff)), 'constant')
            pads[2] = half_diff
        return volume, pads

    def infer_full_volume(self, data, pad):
        # infer full volume with sliding window and overlaps
        data, pads = self.pad_to_atleast(data, 2*pad)
        data = np.pad(data, ((pad//2, pad//2), (pad//2, pad//2), (pad//2, pad//2)))
        vol_out = np.zeros(data.shape + (self.output_channels,), np.float)
        num_x_steps = int(np.ceil(vol_out.shape[0] / pad))
        num_y_steps = int(np.ceil(vol_out.shape[1] / pad))
        num_z_steps = int(np.ceil(vol_out.shape[2] / pad))

        for x in range(num_x_steps):
            for y in range(num_y_steps):
                for z in range(num_z_steps):
                    x0 = x*pad
                    y0 = y*pad
                    z0 = z*pad
                    x0 = np.clip(x0, 0, data.shape[0]-2*pad)
                    y0 = np.clip(y0, 0, data.shape[1]-2*pad)
                    z0 = np.clip(z0, 0, data.shape[2]-2*pad)
                    chunk_in = data[
                        x0:x0+2*pad,
                        y0:y0+2*pad,
                        z0:z0+2*pad,
                        ]
                    net_out = self.infer(chunk_in)
                    vol_out[
                        x0+pad//2:x0+3*pad//2,
                        y0+pad//2:y0+3*pad//2,
                        z0+pad//2:z0+3*pad//2,
                        :
                    ] = net_out[
                            pad//2:3*pad//2,
                            pad//2:3*pad//2,
                            pad//2:3*pad//2,
                            :
                            ]

        vol_out = vol_out[
            pads[0]:vol_out.shape[0]-pads[0],
            pads[1]:vol_out.shape[1]-pads[1],
            pads[2]:vol_out.shape[2]-pads[2],
            :
            ]
        return vol_out[pad//2:-pad//2, pad//2:-pad//2, pad//2:-pad//2, :]

    def infer(self, data):
        data = data[np.newaxis, :, :, :, np.newaxis]
        with torch.no_grad():
            data = torch.from_numpy(data).to(self.device).float().permute(0,4,1,2,3)
            net_out = self.net(data)
            net_out = net_out.permute(0,2,3,4,1).cpu().numpy()
        return net_out[0, :, :, :, :]

class unet_block(torch.nn.Module):
    def __init__(self, num_in, num_out, depth=2, relu=True, norm=True):
        super().__init__()
        depth = max(depth, 1)
        layer_list = self.add_layer(num_in, num_out, relu, norm)
        for _ in range(depth-1):
            layer_list += self.add_layer(num_out, num_out, relu, norm)

        self.layers = torch.nn.Sequential(*layer_list)

    def add_layer(self, num_in, num_out, relu, norm):
        layer_list = []
        conv_layer = torch.nn.Conv3d(num_in, num_out, kernel_size=3, padding=1)
        torch.nn.init.xavier_normal_(conv_layer.weight)
        layer_list.append(conv_layer)
        if norm:
            layer_list.append(torch.nn.BatchNorm3d(num_out))
        if relu:
            layer_list.append(torch.nn.ReLU(inplace=True))
        return layer_list

    def forward(self, x):
        return self.layers(x)

class Unet3D_4(torch.nn.Module):
    def __init__(self, features=8, outputs=1, inputs=1):
        super().__init__()
        self.features = features
        self.outputs = outputs
        self.inputs = inputs

        self.conv_layer_0 = unet_block(self.inputs, self.features)
        self.conv_layer_1 = unet_block(self.features, self.features*2)
        self.conv_layer_2 = unet_block(self.features*2, self.features*4)
        self.conv_layer_3 = unet_block(self.features*4, self.features*8)

        self.conv_layer_4 = unet_block(self.features*8, self.features*8)
        self.conv_layer_5 = unet_block(self.features*16, self.features*4)
        self.conv_layer_6 = unet_block(self.features*8, self.features*2)
        self.conv_layer_7 = unet_block(self.features*4, self.features)

        self.output_layer = unet_block(self.features*2, self.outputs, depth=1, relu=False, norm=False)

        self.maxpool = torch.nn.MaxPool3d(2)
        self.upscale = torch.nn.Upsample(scale_factor=2)

    def forward(self, x): # 64x64xin
        x_0 = self.conv_layer_0(x) # 64x64x8
        x_1 = self.maxpool(x_0) # 32x32x8

        x_1 = self.conv_layer_1(x_1) # 32x32x16
        x_2 = self.maxpool(x_1) # 16x16x16

        x_2 = self.conv_layer_2(x_2) # 16x16x32
        x_3 = self.maxpool(x_2) # 8x8x32

        x_3 = self.conv_layer_3(x_3) # 8x8x64
        x = self.maxpool(x_3) # 4x4x64

        x = self.conv_layer_4(x) # 4x4x64
        x = self.upscale(x) # 8x8x64
        x = torch.cat([x, x_3], dim=1) # 8x8x128

        x = self.conv_layer_5(x) # 8x8x32
        x = self.upscale(x) # 16x16x32
        x = torch.cat([x, x_2], dim=1) # 16x16x64

        x = self.conv_layer_6(x) # 16x16x16
        x = self.upscale(x) # 32x32x16
        x = torch.cat([x, x_1], dim=1) # 32x32x32

        x = self.conv_layer_7(x) # 32x32x8
        x = self.upscale(x) # 64x64x8
        x = torch.cat([x, x_0], dim=1) # 64x64x16

        return self.output_layer(x) # 64x64xout

net_definitions = {
    'UNet_3D_4_16': partial(Unet3D_4, features=16)
}
