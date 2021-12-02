import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, activation_function=nn.ReLU):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            activation_function(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            activation_function(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_function=torch.tanh):
        super(OutConv, self).__init__()
        self.activation_function = activation_function
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.activation_function(self.conv(x))

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation_function=nn.ReLU):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, None, activation_function=activation_function)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, activation_function=nn.ReLU):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, activation_function=activation_function)
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, None, activation_function=activation_function)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        #diffY = x2.size()[2] - x1.size()[2]
        #diffX = x2.size()[3] - x1.size()[3]

        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.reshape(x.shape[0], -1)
        #x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(intermediate_channels * 4),
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

class ResUNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResUNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.doubleconv = DoubleConv(image_channels,self.in_channels)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)
        self.up1 = Up(2048, 1024)
        self.up2 = Up(1024, 512)
        self.up3 = Up(512, 256)
        #self.up4 = Up(256, 128)
        self.outc = OutConv(256, num_classes)

        self.up4 = Up(256, 128)

    def forward(self, input):
        x = input

        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x1 = self.maxpool(x)

        x1 = self.doubleconv(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        #x = self.up4(x, x1)
        x = self.outc(x)

        #x = self.avgpool(x)
        #x = x.reshape(x.shape[0], -1)
        #x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(intermediate_channels * 4),
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(ResBlock, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(ResBlock, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(ResBlock, [3, 8, 36, 3], img_channel, num_classes)

def ResUNet50(img_channel=3, num_classes=1000):
    return ResUNet(ResBlock, [3, 4, 6, 3], img_channel, num_classes)


def ResUNet101(img_channel=3, num_classes=1000):
    return ResUNet(ResBlock, [3, 4, 23, 3], img_channel, num_classes)


def ResUNet152(img_channel=3, num_classes=1000):
    return ResUNet(ResBlock, [3, 8, 36, 3], img_channel, num_classes)

class AAM(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(AAM, self).__init__() 
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, padding=0),
            nn.Softmax(dim=1))

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input_high, input_low):
        mid_high=self.global_pooling(input_high)
        weight_high=self.conv1(mid_high)

        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        weight=self.conv3(weight_low+weight_high)
        low = self.conv4(input_low)
        return input_high+low.mul(weight)

class RAUNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        #assert num_channels == 3
        self.w = 512
        self.h = 640
        self.num_classes = num_classes
        # filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=pretrained)
        filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)
        resnet = ResUNet50(num_channels, num_classes=num_classes)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        self.gau2 = AAM(filters[1], filters[1])
        self.gau1 = AAM(filters[0], filters[0])


        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose1d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv1d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv1d(32, num_classes, 2, padding=1)
        self.tanh = nn.Tanh()

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4)
        b4 = self.gau3(d4, e3)
        d3 = self.decoder3(b4)
        b3 = self.gau2(d3, e2)
        d2 = self.decoder2(b3)
        b2 = self.gau1(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = torch.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return self.tanh(x_out)
        
class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm1d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose1d(in_channels // 4, in_channels // 4, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm1d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv1d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm1d(n_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x
        
