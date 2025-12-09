import torch
import torch.nn as nn
import torch.nn.functional as functional


# -------------- fixed architecture Encoder+Decoder -----------------
class Encoder(nn.Module):
    def __init__(self, in_chan, num_latent=16, kernel_size=5, stride=2, padding=(1,2,1,2)):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_chan, 32, kernel_size, stride=stride, padding=padding), nn.ReLU(True),
                                  nn.Conv2d(32, 64, kernel_size, stride=stride, padding=padding), nn.ReLU(True),
                                  nn.Conv2d(64, 256, kernel_size, stride=stride, padding=padding), nn.ReLU(True),
                                  nn.Conv2d(256, 512, kernel_size, stride=stride, padding=padding), nn.ReLU(True)
                                  )

        self.fc_en = nn.Linear(512 * 8 * 8, num_latent)

    def forward(self, x):
        x_conv = self.conv(x)
        return self.fc_en(x_conv.view(x.size(0), -1))


class Decoder(nn.Module):
    def __init__(self, out_chan=2, num_latent=16, kernel_size=5, stride=2, padding=(1,2,1,2)):
        super().__init__()
        self.fc_dec = nn.Sequential(nn.Linear(num_latent, 64), nn.ReLU(True))

        self.deconv = nn.Sequential(nn.ConvTranspose2d(1, 256, kernel_size, stride=stride, padding=padding), nn.ReLU(True),
                                    nn.ConvTranspose2d(256, 64, kernel_size, stride=stride, padding=padding), nn.ReLU(True),
                                    nn.ConvTranspose2d(64, 32, kernel_size, stride=stride, padding=padding), nn.ReLU(True),
                                    nn.ConvTranspose2d(32, out_chan, kernel_size, stride=stride, padding=padding)
                                    )

    def forward(self, x):
        x_fc = self.fc_dec(x).view(x.size(0), 1, 8, 8)
        return self.deconv(x_fc)


# ------------------- flexible architecture Encoder+Decoder ------------------
class FlexibleEncoder(nn.Module):
    def __init__(self, hparams):
        """
        Parameters
        ----------
        hparams : :obj:`dict`
            -  input_dim : :obj:`array-like`
                    dimensions of image with shape (n_channels, H_pix, W_pix)
            - 'n_latents' (:obj:`int`)
            - 'channels' (:obj:`list`)
            - 'kernel_size' (:obj:`int`)
            - 'stride_size' (:obj:`int`)
            - 'padding' (:obj:`list`)
            - 'output_size' (:obj:`list`)
        """

        super().__init__()
        self.hparams = hparams
        self.encoder = None
        self.fc = None
        self.build_model()

    def build_model(self):
        """
        construct the encoder model
        """

        self.encoder = nn.ModuleList()

        for i_layer in range(len(self.hparams['channels'])):
            args = self.get_layer_params(i_layer)
            # add conv module
            module = nn.Conv2d(
                in_channels=args['in_channels'],
                out_channels=args['out_channels'],
                kernel_size=args['kernel_size'],
                stride=args['stride'],
                padding=args['padding'])
            self.encoder.add_module(f'Conv2d - layer {i_layer+1}', module)
            # add RELU module
            module = nn.ReLU(True)
            self.encoder.add_module(f'ReLu - layer {i_layer + 1}', module)  # in case of slow convergence or many dead filters -> switch to Leaky ReLU (alpha = 0.01)
            # if stride = 1 add max pool for spatial downsampling
            if args['stride'] == 1:
                module = nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                    return_indices=True,
                )
                self.encoder.add_module(f'Max Pool - layer {i_layer + 1}', module)

        # flatten resulting spatial size
        linear_dim = self.hparams['channels'][-1] * self.hparams['output_dim'][-1] * self.hparams['output_dim'][-1]
        # final fully connected layer
        self.fc = nn.Linear(linear_dim,  self.hparams['num_latent'])


    def get_layer_params(self, i_layer):
        if i_layer == 0:
            in_channels = self.hparams['input_dim'][0]
        else:
            in_channels = self.hparams['channels'][i_layer-1]

        out_channels = self.hparams['channels'][i_layer]
        left_pad = self.hparams['padding'][i_layer][0]
        right_pad = self.hparams['padding'][i_layer][1]
        if left_pad == right_pad:
            # symmetric padding
            padding = left_pad
        else:
            # asymmetric padding - need to add a special padding module before the conv layer
            module = nn.ZeroPad2d((left_pad, right_pad, left_pad, right_pad))
            self.encoder.add_module(f'Zero Padding - layer {i_layer+1}', module)
            padding = 0

        args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': self.hparams['kernel'],
            'stride': self.hparams['stride'],
            'padding': padding,
        }

        return args

    def forward(self, x):
        # x = self.encoder(x)

        # loop over layers, have to collect pool_idx and output sizes if using max pooling to use
        # in unpooling
        pool_idx = []
        target_output_size = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                target_output_size.append(x.size())
                x, idx = layer(x)
                pool_idx.append(idx)
            else:
                x = layer(x)

        # flatten tensor for the fully connected final layer
        x = x.view(x.size(0), -1)
        z = self.fc(x)

        #return z, pool_idx, target_output_size
        return z


class FlexibleDecoder(nn.Module):
    def __init__(self, hparams):

        """
        Parameters
        ----------
        hparams : :obj:`dict`
            -  input_dim : :obj:`array-like`
                    dimensions of image with shape (n_channels, H_pix, W_pix)
            - 'n_latents' (:obj:`int`)
            - 'channels' (:obj:`list`)
            - 'kernel_size' (:obj:`int`)
            - 'stride_size' (:obj:`int`)
            - 'padding' (:obj:`list`)
            - 'output_dim' (:obj:`list`)
        """

        super().__init__()
        self.hparams = hparams
        self.decoder = None
        self.fc_dec = None
        self.conv_t_pads = {}
        self.build_model()

    def build_model(self):
        """
        construct the encoder model
        """

        # fc layer to restore flattened feature map
        linear_dim = self.hparams['channels'][-1] * self.hparams['output_dim'][-1] * self.hparams['output_dim'][-1]
        self.fc_dec = nn.Sequential(
            nn.Linear(self.hparams['num_latent'], linear_dim),
            nn.ReLU(True)
        )

        self.decoder = nn.ModuleList()

        # build deconv stack in reverse order
        reversed_channels = list(reversed(self.hparams['channels']))
        reversed_channels.append( self.hparams['input_dim'][0])

        for i_layer in range(len(self.hparams['channels'])):
            args = self.get_layer_params(i_layer, reversed_channels)

            # if stride = 1 add max Unpool for spatial upsampling
            # if args['stride'] == 1:
            #     module = nn.MaxUnPool2d(
            #         kernel_size=2,
            #         stride=2
            #     )
            #
            #     self.decoder.add_module(f'Max UnPool - layer {i_layer + 1}', module)

            # add conv transpose module
            module = nn.ConvTranspose2d(
                in_channels=args['in_channels'],
                out_channels=args['out_channels'],
                kernel_size=args['kernel_size'],
                stride=args['stride'],
                padding=args['padding'],
                output_padding=args['output_padding']
            )
            self.decoder.add_module(f'Convtranspose2d - layer {i_layer + 1}', module)

            # add RELU module
            if i_layer < len(self.hparams['channels']) - 1:  # last layer has no ReLU
                module = nn.ReLU(True)
                self.decoder.add_module(f'ReLu - layer {i_layer + 1}', module)  # in case of slow convergence or many dead filters -> switch to Leaky ReLU (alpha = 0.01)
            # last layer has tanh module to get values in [-1,1]
            else:
                #module = nn.Tanh()
                module = nn.Sigmoid()
                self.decoder.add_module(f'Tanh - layer {i_layer + 1}', module)


    def get_layer_params(self, i_layer, reversed_channels):
        in_channels = reversed_channels[i_layer]
        out_channels = reversed_channels[i_layer + 1]
        kernel = self.hparams['kernel']
        stride = self.hparams['stride'] if self.hparams['stride'] > 1 else 2
        num_paddings = len(self.hparams['padding'])
        left_pad = self.hparams['padding'][num_paddings - 1 - i_layer][0]
        right_pad = self.hparams['padding'][num_paddings - 1 - i_layer][1]
        output_padding = 0
        dim_num = len(self.hparams['output_dim'])
        in_dim = self.hparams['output_dim'][dim_num - 1 - i_layer]
        desired_out_dim = self.hparams['output_dim'][dim_num - 2 - i_layer]

        if left_pad == right_pad:
            # symmetric padding
            padding = left_pad
            # compute output dim
            calc_out = (in_dim - 1) * stride - 2 * padding + kernel

            self.conv_t_pads[f'Convtranspose2d - layer {i_layer + 1}'] = None

        else:
            # asymmetric padding - need to add a special padding module after the conv layer (cut extra pixels) in forward()
            padding = 0
            output_padding = 0
            # compute output dim
            calc_out = (in_dim - 1) * stride - (left_pad + right_pad) + kernel

            self.conv_t_pads[f'Convtranspose2d - layer {i_layer + 1}'] = [left_pad, right_pad, left_pad, right_pad]

        # compute output padding
        if calc_out < desired_out_dim:
            output_padding = 1
            if abs(desired_out_dim - calc_out) > 1:
                raise ValueError('the output dimension of decoder model is wrong - check parameters')

        args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': self.hparams['kernel'],
            'stride': self.hparams['stride'] if self.hparams['stride'] > 1 else 2,  # if stride=1 -> use max-Unpooling or convtranspose with stride=2
            'padding': padding,
            'output_padding':output_padding,
        }

        return args

    def forward(self, z):
        x = self.fc_dec(z)
        x = x.view(z.size(0), self.hparams['channels'][-1], self.hparams['output_dim'][-1], self.hparams['output_dim'][-1])  # symmetric reshape

        for name, layer in self.decoder.named_children():
            # if isinstance(layer, nn.MaxUnpool2d):
            #     idx = pool_idx.pop(-1)
            #     outsize = target_output_size.pop(-1)
            #     x = layer(x, idx, outsize)
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                if self.conv_t_pads[name] is not None:
                    # asymmetric padding for convtranspose layer if necessary
                    # (-i does cropping!)
                    x = functional.pad(x, [-i for i in self.conv_t_pads[name]])

            else:
                x = layer(x)

        return x





