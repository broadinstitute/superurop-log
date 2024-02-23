import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

backbone = torchvision.models.resnet50(pretrained=True)


############### Feature Pyramid Network ###############
class FPN(nn.Module):
    """
    Implemented FPN here is different from the FPN introduced in https://arxiv.org/pdf/1612.03144.pdf.
    """

    def __init__(self, in_channels):
        '''
        in_channels=[512, 1024, 2048]
        '''
        super().__init__()
        self.num_downsample = 2
        self.in_channels = in_channels

        self.last_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in reversed(self.in_channels)])
        # 1 x 1 conv to backbone feature map
        # ModuleList((0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        #            (1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        #            (2): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)))

        self.final_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, padding=1) for _ in self.in_channels])
        # 3 x 3 conv to FPN feature map in order to recover error that might be occur during upsampling 
        # and add two different feature map
        # ModuleList((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #            (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        self.downsample_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
                                                for _ in range(self.num_downsample)])
        # 3 x 3 conv to P5 in order to make P6, P7 final feature map
        # ModuleList((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

    def forward(self, backbone_outs):
        '''
        #backbone_outs = [[n, 512, 69, 69], [n, 1024, 35, 35], [n, 2048, 18, 18]]
        In class Yolact's train(), remove C2 from bakebone_outs. So FPN gets three feature outs.
        '''
        out = []
        x = torch.zeros(1, device=backbone_outs[0].device)
        for i in range(len(backbone_outs)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(backbone_outs)  # convouts: C3, C4, C5

        for last_layer in self.last_layers:
            j -= 1
            if j < len(backbone_outs) - 1:
                #backbone_outs = [[n, 512, 69, 69], [n, 1024, 35, 35], [n, 2048, 18, 18]]
                _, _, h, w = backbone_outs[j].size()
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            x = x + last_layer(backbone_outs[j])
            out[j] = x
        j = len(backbone_outs)
        for final_layer in self.final_layers:
            j -= 1
            out[j] = F.relu(final_layer(out[j]))

        for layer in self.downsample_layers:
            out.append(layer(out[-1]))

        return out




############### Proto Network ###############
'''
Use P3 which is deepest feature map and has highest resolution
'''
mask_proto_net = [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}),
                  (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})]

class Protonet(nn.Module) :
    def __init__(self, mask_proto_net) :
        super().__init__()

        self.inplanes=256
        self.mask_proto_net = mask_proto_net
        self.conv1 = nn.Conv2d(self.inplanes, mask_proto_net[0][0], kernel_size=mask_proto_net[0][1], **mask_proto_net[0][2])
        self.conv2 = nn.Conv2d(self.inplanes, mask_proto_net[1][0], kernel_size=mask_proto_net[1][1], **mask_proto_net[1][2])
        self.conv3 = nn.Conv2d(self.inplanes, mask_proto_net[2][0], kernel_size=mask_proto_net[2][1], **mask_proto_net[2][2])
        self.conv4 = nn.Conv2d(self.inplanes, mask_proto_net[4][0], kernel_size=mask_proto_net[4][1], **mask_proto_net[4][2])
        self.conv5 = nn.Conv2d(self.inplanes, mask_proto_net[5][0], kernel_size=mask_proto_net[5][1], **mask_proto_net[5][2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor = -self.mask_proto_net[3][1], mode='bilinear', align_corners=False, **self.mask_proto_net[3][2])
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        
        
        return out

#proto_out : [n, 32, 138, 138]
coef_dim=proto_out.shape[1]
num_classes=81
aspect_ratios: [1, 1 / 2, 2]
class PredictionModule(nn.Module):
    def __init__(self, in_channels, coef_dim):
        super().__init__()

        self.num_classes = 81
        self.coef_dim = coef_dim
        self.num_priors = 3            # num of anchor box for each pixel of feature map

        self.upfeature = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        out_channels = 256
        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.coef_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upfeature(x)
        x = self.relu(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        bbox = self.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        coef_test = self.mask_layer(x)
        print('mask layer output shape : ', coef_test.shape)
        coef = self.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.coef_dim)       
        # mask_layer output shape : [n, 96, 69, 69] / In order to make it's shape [n, 69*69*3, 32], use permute and contiguous.
        print('Changed shape : ', coef.shape)
        coef = torch.tanh(coef)

        return {'box': bbox, 'class': conf, 'coef': coef}
prediction_layers = nn.ModuleList()
prediction_layers.append(PredictionModule(in_channels=256, coef_dim=coef_dim))

