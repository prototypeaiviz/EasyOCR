import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models
from collections import namedtuple
from packaging import version

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT INITIALISATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def init_weights(modules):
    # Applies sensible default initialisations to each layer type:
    #   Conv2d   → Xavier uniform (keeps variance stable across layers)
    #   BatchNorm→ weight=1, bias=0 (identity at start of training)
    #   Linear   → small normal (0.01 std), bias=0
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

# ─────────────────────────────────────────────────────────────────────────────
# VGG-16 WITH BATCH NORM  (used as the DETECTION backbone, not recognition)
# ─────────────────────────────────────────────────────────────────────────────

class vgg16_bn(torch.nn.Module):
    # Slices pretrained VGG-16 BN into 5 feature stages for use in CRAFT
    # (the text DETECTION model).  Not part of the recognition pipeline.

    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else:
            models.vgg.model_urls['vgg16_bn'] = models.vgg.model_urls['vgg16_bn'].replace('https://', 'http://')
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        # Split VGG features into 5 named slices so each intermediate activation
        # (relu2_2, relu3_2, relu4_3, relu5_3, fc7) can be accessed separately.
        # CRAFT uses multi-scale features from all 5 levels.
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # layers 0-11  → up to conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):     # layers 12-18 → up to conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):     # layers 19-28 → up to conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):     # layers 29-38 → up to conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # Replaces VGG's fully-connected layers fc6/fc7 with dilated convolutions
        # (dilation=6 on a 3×3 kernel sees a 13×13 receptive field without pooling),
        # allowing larger spatial context while keeping the feature map resolution.
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  # slice5 has no pretrained weights

        if freeze:
            # Keep early low-level filters (edges, colours) fixed; only fine-tune
            # deeper layers that encode higher-level text features.
            for param in self.slice1.parameters():
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

# ─────────────────────────────────────────────────────────────────────────────
# BIDIRECTIONAL LSTM  (shared by both generation1 and generation2)
# ─────────────────────────────────────────────────────────────────────────────

class BidirectionalLSTM(nn.Module):
    # Wraps PyTorch's built-in bidirectional LSTM and adds a linear projection.
    #
    # Why bidirectional?
    #   A standard (unidirectional) LSTM at position t only sees columns 0..t.
    #   For OCR, knowing the next character often helps disambiguate the current
    #   one (e.g. distinguishing 'rn' from 'm').  A backward LSTM reads the
    #   sequence in reverse, so at position t it has seen t..T-1.  Their hidden
    #   states are concatenated, giving full context in both directions.
    #
    # Two BiLSTMs are stacked:
    #   Layer 1: 512-d CNN features → hidden_size-d context
    #   Layer 2: hidden_size-d      → hidden_size-d  (deeper temporal reasoning)

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        # bidirectional=True doubles the output: forward_hidden ++ backward_hidden
        # batch_first=True: input/output shape is [B, T, features] (more intuitive)
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # Project 2*hidden_size → output_size to keep dimensions consistent
        # between stacked layers (both input and output are output_size).
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # input:  [B, T, input_size]
        # output: [B, T, output_size]

        try:
            # flatten_parameters() makes the LSTM weights contiguous in memory,
            # which is required for multi-GPU (DataParallel) to work.
            self.rnn.flatten_parameters()
        except:
            # Dynamic quantisation replaces the LSTM internals, so
            # flatten_parameters() is no longer available.
            pass

        recurrent, _ = self.rnn(input)
        # recurrent: [B, T, 2*hidden_size]
        # The second return value is (h_n, c_n) — final hidden/cell states,
        # not needed here since we want all time-step outputs.

        output = self.linear(recurrent)
        # output: [B, T, output_size]
        return output

# ─────────────────────────────────────────────────────────────────────────────
# VGG FEATURE EXTRACTOR  (generation2 backbone)
# ─────────────────────────────────────────────────────────────────────────────

class VGG_FeatureExtractor(nn.Module):
    # Lightweight VGG-style CNN for text feature extraction.
    # Uses MaxPool layers with asymmetric strides to reduce H faster than W,
    # preserving the horizontal extent of the text for sequence modelling.
    #
    # With input [B, 1, 32, W] and output_channel=256:
    #   Channel progression: 1 → 32 → 64 → 128 → 128 → 256 → 256 → 256
    #   Spatial:  H goes 32→16→8→4→2→1,  W is mostly preserved

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        # Split output_channel into 4 levels: /8, /4, /2, /1
        # e.g. for output_channel=256: [32, 64, 128, 256]
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            # Block 1: conv 3×3 → ReLU → 2×2 pool (halves both H and W)
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                         # H/2, W/2

            # Block 2: conv 3×3 → ReLU → 2×2 pool
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                         # H/2, W/2

            # Block 3: two conv layers, then asymmetric pool (2×1, stride 2×1)
            # — halves H but leaves W unchanged, so the horizontal sequence is preserved
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                               # H/2, W unchanged

            # Block 4: two conv layers with BatchNorm (stabilises training),
            # then another asymmetric pool
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                               # H/2, W unchanged

            # Final 2×1 conv (no padding) to squeeze out the last height pixel
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True)
            # Output: [B, output_channel, 1, W']
        )

    def forward(self, input):
        return self.ConvNet(input)

# ─────────────────────────────────────────────────────────────────────────────
# RESNET FEATURE EXTRACTOR  (generation1 backbone)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet_FeatureExtractor(nn.Module):
    # Thin wrapper that instantiates the custom ResNet below.
    # layers=[1,2,5,3] means: stage1 has 1 BasicBlock, stage2 has 2, etc.

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)

# ─────────────────────────────────────────────────────────────────────────────
# BASIC RESIDUAL BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    # Standard residual block from He et al. (ResNet paper):
    #
    #   input ──► conv3×3 → BN → ReLU → conv3×3 → BN ──► + ──► ReLU ──► output
    #        └──────────────────────────────────────────►┘
    #                   skip (identity or 1×1 conv)
    #
    # The skip connection lets gradients flow directly back through the network,
    # solving the vanishing gradient problem in deep networks.
    # If in/out channels differ, a 1×1 conv (downsample) projects the skip.

    expansion = 1   # BasicBlock keeps the same channel count (vs Bottleneck which expands)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)   # inplace saves memory
        self.downsample = downsample        # optional projection for skip
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        # padding=1 keeps spatial size unchanged (for stride=1)
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
        # bias=False because BatchNorm already has a learnable bias term

    def forward(self, x):
        residual = x                    # save input for skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)             # BN before add (pre-activation variant would do it after)

        if self.downsample is not None:
            residual = self.downsample(x)   # match channels/spatial size to `out`

        out += residual                 # skip connection: element-wise addition
        out = self.relu(out)

        return out

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM RESNET  (generation1 backbone)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet(nn.Module):
    # Custom ResNet tailored for text recognition:
    #   - Starts with two 3×3 conv layers instead of a single 7×7 conv.
    #   - Uses asymmetric pooling (stride (2,1)) to preserve width.
    #   - Ends with two special convolutions to bring H all the way down to 1.
    #
    # Channel schedule with output_channel=512:
    #   stem:   1 → 32 → 64
    #   stage1: 64 → 128   (1 block)
    #   stage2: 128 → 256  (2 blocks)
    #   stage3: 256 → 512  (5 blocks)
    #   stage4: 512 → 512  (3 blocks)
    #
    # Spatial schedule for H=32, W=100:
    #   after stem+pool1: H=16, W=50
    #   after pool2:      H=8,  W=25
    #   after pool3:      H=4,  W=26  (asymmetric: H/2, W+1 due to padding)
    #   after conv4_1:    H=2,  W=27
    #   after conv4_2:    H=1,  W=26
    # → final shape: [B, 512, 1, ~26]

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        # Channel sizes for each stage: [128, 256, 512, 512]
        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        # Initial number of channels after the stem (= output_channel/8 = 64)
        self.inplanes = int(output_channel / 8)

        # ── Stem: two 3×3 convs (replaces the standard ResNet 7×7 conv) ─────
        # Smaller kernels give finer control and are easier to optimise.
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        # input_channel=1 → output_channel/16=32 channels
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))

        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        # 32 → 64 channels
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)

        self.relu = nn.ReLU(inplace=True)

        # ── Stage 1: 1 BasicBlock, 128 channels ──────────────────────────────
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # H/2, W/2
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        # Extra conv after residual blocks (without stride) to deepen features
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        # ── Stage 2: 2 BasicBlocks, 256 channels ─────────────────────────────
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # H/2, W/2
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        # ── Stage 3: 5 BasicBlocks, 512 channels ─────────────────────────────
        # Asymmetric pool: stride (2,1) halves H but leaves W unchanged.
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))  # H/2, W ~same
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        # ── Stage 4: 3 BasicBlocks, 512 channels ─────────────────────────────
        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)

        # Two final convolutions that bring H from ~4 → 2 → 1:
        # conv4_1: kernel (2,2), stride (2,1), padding (0,1) → H/2, W ~same
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                 kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])

        # conv4_2: kernel (2,2), stride 1, no padding → reduces H from 2 to 1
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                 kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        # Creates a sequential stage of `blocks` BasicBlocks.
        # The first block may downsample (via a 1×1 conv) if channel counts change.
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 1×1 conv to match dimensions for the skip connection
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion    # update running channel count
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # remaining blocks: same in/out

        return nn.Sequential(*layers)

    def forward(self, x):
        # ── Stem ─────────────────────────────────────────────────────────────
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        # shape: [B, 64, 32, W]

        # ── Stage 1 ──────────────────────────────────────────────────────────
        x = self.maxpool1(x)            # [B, 64, 16, W/2]
        x = self.layer1(x)              # [B, 128, 16, W/2]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        x = self.maxpool2(x)            # [B, 128, 8, W/4]
        x = self.layer2(x)              # [B, 256, 8, W/4]
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # ── Stage 3 ──────────────────────────────────────────────────────────
        x = self.maxpool3(x)            # [B, 256, 4, ~W/4]  (asymmetric: H only)
        x = self.layer3(x)              # [B, 512, 4, ~W/4]
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # ── Stage 4 ──────────────────────────────────────────────────────────
        x = self.layer4(x)              # [B, 512, 4, ~W/4]
        x = self.conv4_1(x)             # [B, 512, 2, ~W/4]  (H: 4→2)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)             # [B, 512, 1, ~W/4]  (H: 2→1)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x
        # Final shape: [B, 512, 1, W']
        # Ready for AdaptiveAvgPool + squeeze → [B, W', 512] sequence
