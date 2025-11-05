import torch
import torch.nn as nn



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
    def __init__(
        self,
        in_chan,
        num_latent,
        H_input,
        W_input,
        channels=(32, 64, 128, 256),
        kernel_size=5,
        stride=2,
        padding=(1, 2),
        use_maxpool=False,
    ):
        super().__init__()

        self.channels = list(channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_maxpool = use_maxpool

        layers = []
        prev_ch = in_chan

        for ch in self.channels:
            layers.append(nn.Conv2d(prev_ch, ch, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU(True))
            if use_maxpool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = ch

        self.conv = nn.Sequential(*layers)

        # --- automatically compute resulting spatial size ---
        with torch.no_grad():
            dummy = torch.zeros(1, in_chan, H_input, W_input)
            out = self.conv(dummy)
            self.C_out = out.shape[1]
            self.H_out = out.shape[2]
            self.W_out = out.shape[3]
            self.feat_dim = out.numel()

        self.fc_en = nn.Linear(self.feat_dim, num_latent)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc_en(x)
        return z


class FlexibleDecoder(nn.Module):
    def __init__(
        self,
        out_chan,
        num_latent,
        channels=(32, 64, 128, 256),
        kernel_size=5,
        stride=2,
        padding=(1, 2),
        use_maxpool=False,
    ):

        super().__init__()
        self.out_chan = out_chan
        self.num_latent = num_latent
        self.channels = list(channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_maxpool = use_maxpool

        # Shapes will be filled from encoder
        self.C_out = None
        self.H_out = None
        self.W_out = None
        self.feat_dim = None

        self.fc_dec = None
        self.deconv = None

    def build_from_encoder(self, encoder: FlexibleEncoder):
        self.C_out = encoder.C_out
        self.H_out = encoder.H_out
        self.W_out = encoder.W_out
        self.feat_dim = encoder.feat_dim

        # fc layer to restore flattened feature map
        self.fc_dec = nn.Sequential(
            nn.Linear(self.num_latent, self.feat_dim),
            nn.ReLU(True)
        )

        # Build deconv stack in reverse order
        deconvs = []
        reversed_channels = list(reversed(self.channels))

        prev_ch = reversed_channels[0]
        for i, ch in enumerate(reversed(self.channels[1:] + [self.out_chan])):
            # if encoder used maxpool - upsample by 2
            s = 2 if self.use_maxpool else self.stride
            op = s - 1

            deconvs.append(nn.ConvTranspose2d(
                prev_ch, ch,
                self.kernel_size,
                stride=s,
                padding=self.padding,
                output_padding=op
            ))
            if i < len(self.channels):  # last layer has no ReLU
                deconvs.append(nn.ReLU(True))
            prev_ch = ch

        self.deconv = nn.Sequential(*deconvs)

    def forward(self, z):
        x = self.fc_dec(z)
        x = x.view(z.size(0), self.C_out, self.H_out, self.W_out)  # symmetric reshape
        x = self.deconv(x)
        return x





