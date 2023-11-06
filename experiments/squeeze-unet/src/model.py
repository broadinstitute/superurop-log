"""
Paper: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w22/Beheshti_Squeeze_U-Net_A_Memory_and_Energy_Efficient_Image_Segmentation_Network_CVPRW_2020_paper.pdf
"""

import torch.nn as nn


class SqueezeUnet(nn.Module):

    def __init__(self):
        super(SqueezeUnet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.squeeze_excitation = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.squeeze_excitation(x)
        x = self.decoder(x)

        return x
