from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import unbatch

import einops


class UNet1d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, pos_dim=2):
        super(UNet1d, self).__init__()

        features = init_features
        self.encoder1 = UNet1d._block(in_channels + pos_dim, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def padding_shape(self, x):
        if x.shape[-1] % 2 == 1:
            x = torch.cat([x, torch.zeros_like(x[..., 0:1])], dim=-1)
        return x
    
    def padding_fit(self, x, y):
        if x.shape[-1] < y.shape[-1]:
            x = torch.cat([x, torch.zeros_like(x[..., 0:y.shape[-1] - x.shape[-1]])], dim=-1)
        elif x.shape[-1] > y.shape[-1]:
            y = torch.cat([y, torch.zeros_like(y[..., 0:x.shape[-1] - y.shape[-1]])], dim=-1)
        return x, y
    
    def forward(self, data):
        """

        Args:
            x (torch.Tensor): (batch_size, t_grid, x_grid, in_channels)

        Returns:
            x (torch.Tensor): (batch_size, t_grid, x_grid, out_channels)
        """
        x = torch.cat([data.x, data.pos], dim=-1)
        x = pad_sequence(unbatch(x, data.batch), batch_first=True, padding_value=0)
        x = einops.rearrange(x, 'b n c -> b c n')

        x = self.padding_shape(x)
        enc1 = self.encoder1(x)
        enc1 = self.padding_shape(enc1)
        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.padding_shape(enc2)
        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.padding_shape(enc3)
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4, dec4 = self.padding_fit(enc4, dec4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        enc3, dec3 = self.padding_fit(enc3, dec3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2, dec2 = self.padding_fit(enc2, dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        enc1, dec1 = self.padding_fit(enc1, dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.conv(dec1)
        out = einops.rearrange(out, 'b c n -> b n c')
        out = torch.cat([out[i, :batch.shape[0]] for i, batch in enumerate(unbatch(data.batch, data.batch))], dim=0)
        
        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
