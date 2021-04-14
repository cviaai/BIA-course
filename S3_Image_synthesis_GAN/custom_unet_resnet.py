import torch
import torch.nn as nn
import torch.nn.functional as F

activations = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'softmax': F.softmax
}


# Some basic blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu'):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.activation != 'linear':
            x = activations[self.activation](x)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu'):
        super(ConvTransposeBlock, self).__init__()
        self.activation = activation
        self.conv_tr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                          padding, bias=bias)

    def forward(self, x):
        x = self.conv_tr(x)
        if self.activation != 'linear':
            x = activations[self.activation](x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, isDownsample=False,
                 isEqDecoder=False, normalization='batch', downsample_type='max'):
        super(ResBlock, self).__init__()

        self.isDownsample = isDownsample
        self.isEqDecoder = isEqDecoder
        if self.isDownsample:
            if downsample_type == 'max':
                self.downsample = nn.MaxPool2d(2, 2, 0)
            elif downsample_type == 'avg':
                self.downsample = nn.AvgPool2d(2, 2, 0)
            else:
                self.downsample = nn.Conv2d(in_channels, in_channels, 2, 2)
            self.equalize = nn.Conv2d(in_channels, out_channels, 2, 2, 0, bias=bias)

        if self.isEqDecoder:
            self.eq_dec = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if normalization == 'batch':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        identity = x
        if self.isDownsample:
            out = self.downsample(x)
        else:
            out = x
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        if self.isDownsample:
            out = self.equalize(identity) + out
        elif self.isEqDecoder:
            out = self.eq_dec(identity) + out
        else:
            out = identity + out
        out = F.relu(out)
        return out


# U-net blocks
class DecoderBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_sizes, strides, paddings, bias):
        super(DecoderBlock, self).__init__()
        self.decoder_blocks = []
        self.conv_transpose = ConvTransposeBlock(in_features, out_features, kernel_sizes[0], strides[0], paddings[0],
                                                 bias)
        for i in range(len(kernel_sizes[1:])):
            self.decoder_blocks.append(
                ConvBlock(in_features, out_features, kernel_sizes[i + 1], strides[i + 1], paddings[i + 1],
                          bias))
            in_features = out_features
        self.res_blocks = nn.Sequential(*self.decoder_blocks)

    def forward(self, lower_dim, higher_dim):
        upsample = self.conv_transpose(lower_dim)
        res = self.res_blocks(torch.cat((upsample, higher_dim), 1))
        return res


class UnetResnetEncoder(nn.Module):
    def __init__(self, in_channels, start_features_num, expand_rate, kernel_sizes,
                 strides, paddings, bias, blocks_in_layer, normalization, downsample_type):
        super(UnetResnetEncoder, self).__init__()
        self.encoder_blocks = []
        features_num = start_features_num
        self.blocks_in_layer = blocks_in_layer
        self.encoder_blocks.append(ConvBlock(in_channels, features_num, kernel_sizes[0], strides[0],
                                             paddings[0], bias))
        for i in range(1, len(kernel_sizes)):
            in_channels = features_num
            if i % blocks_in_layer == 0:
                isDownsample = True
                features_num *= expand_rate
            else:
                isDownsample = False
            self.encoder_blocks.append(ResBlock(in_channels, features_num, kernel_sizes[i], strides[i],
                                                paddings[i], bias, isDownsample, False, normalization, downsample_type))
        self.encoder_blocks = nn.Sequential(*self.encoder_blocks)

    def forward(self, x):
        encoder_passes = []
        for idx, encoder_block in enumerate(self.encoder_blocks.children()):
            x = encoder_block(x)
            if idx % self.blocks_in_layer == 1:
                encoder_passes.append(x)
        return encoder_passes


class UnetDecoder(nn.Module):
    def __init__(self, start_features_num, expand_rate, kernel_sizes,
                 decoder_kernels, bias, blocks_in_layer, mode, normalization, downsample_type):
        super(UnetDecoder, self).__init__()
        self.mode = mode
        self.decoder_blocks = []
        self.num_downsampling = len(kernel_sizes) // blocks_in_layer - 1
        max_features = start_features_num * (expand_rate ** self.num_downsampling)
        strides = [2] + [1] * len(decoder_kernels[1:])
        paddings = [0] + list(map(lambda x: x // 2, decoder_kernels[1:]))
        for i in range(self.num_downsampling):
            decoder_block = DecoderBlock(max_features, max_features // expand_rate, decoder_kernels, strides, paddings,
                                         bias)
            max_features = max_features // expand_rate
            self.decoder_blocks.append(decoder_block)
        self.decoder_blocks = nn.Sequential(*self.decoder_blocks)

    def forward(self, encoder_passes):
        x = encoder_passes[-1]
        for idx, decoder_block in enumerate(self.decoder_blocks.children()):
            x = decoder_block(x, encoder_passes[-(idx + 2)])
        return x


class CustomUnetResnet(nn.Module):
    def __init__(self, in_channels, out_channels, start_features_num, expand_rate,
                 kernel_sizes, decoder_kernels, bias, blocks_in_layer,
                 final_activation, mode, normalization, downsample_type):
        super(CustomUnetResnet, self).__init__()
        paddings = list(map(lambda x: x // 2, kernel_sizes))
        strides = [1] * len(paddings)
        self.encoder = []
        self.decoder_blocks = []
        self.mode = mode
        self.encoder = UnetResnetEncoder(in_channels, start_features_num, expand_rate, kernel_sizes,
                                         strides, paddings, bias, blocks_in_layer, normalization, downsample_type)

        self.decoder = UnetDecoder(start_features_num, expand_rate, kernel_sizes, decoder_kernels,
                                   bias, blocks_in_layer, mode, normalization, downsample_type)

        self.final_conv = ResBlock(start_features_num, start_features_num, kernel_sizes[-1],
                                   strides[-1], paddings[-1], bias, False, False, normalization, downsample_type)
        self.final_conv2 = ConvBlock(start_features_num, out_channels, kernel_sizes[-1],
                                     strides[-1], paddings[-1], bias, final_activation)

    def forward(self, x):
        encoder_passes = self.encoder(x)
        out = self.decoder(encoder_passes)
        out = self.final_conv(out)
        out = self.final_conv2(out)
        return out


def get_custom_unet_resnet(in_channels, out_channels, start_features_num, expand_rate, kernel_sizes, decoder_kernels,
                           bias, blocks_in_layer, final_activation, mode, normalization, downsample_type):
    return CustomUnetResnet(**locals())
