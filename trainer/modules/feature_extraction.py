import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# VGG FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
# Based on the original CRNN paper (Shi et al., 2015).
# A stack of Conv→ReLU blocks with MaxPool layers whose strides are chosen to
# collapse image height (H) much faster than width (W), so the output is a
# wide, flat feature map that can be read as a left-to-right sequence.
#
# With output_channel=512 and input [B, 1, 32, 100]:
#   channel schedule: 1 → 64 → 128 → 256 → 256 → 512 → 512 → 512
#   spatial schedule (H × W):
#     after pool1 (2×2): 16 × 50
#     after pool2 (2×2):  8 × 25
#     after pool3 (2×1):  4 × 25   ← asymmetric: H/2 but W unchanged
#     after pool4 (2×1):  2 × 25
#     after final conv:   1 × 24   ← H collapses to 1

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            # Block 1: 3×3 conv → ReLU → 2×2 pool  (H/2, W/2)
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                          # → 64×16×50

            # Block 2: 3×3 conv → ReLU → 2×2 pool  (H/2, W/2)
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                          # → 128×8×25

            # Block 3: two 3×3 convs, then asymmetric pool (2×1, stride 2×1)
            # The asymmetric kernel/stride halves H but leaves W alone.
            # This is the key design choice: horizontal extent = character positions.
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # → 256×8×25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                                # → 256×4×25

            # Block 4: two 3×3 convs with BatchNorm (stabilises training at depth),
            # then another asymmetric pool
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),       # → 512×4×25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                                # → 512×2×25

            # Final 2×1 conv (no padding): squeezes remaining H from 2 → 1
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True)
            # → 512×1×24
        )

    def forward(self, input):
        return self.ConvNet(input)  # [B, 512, 1, ~W/4]


# ─────────────────────────────────────────────────────────────────────────────
# RCNN (GATED RECURRENT CNN) FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
# Based on GRCNN (Wang et al., NeurIPS 2017).
# Same spatial schedule as VGG but replaces plain Conv blocks with GRCL units,
# which apply the same convolution weights recurrently (num_iteration times)
# and gate each recurrent update.  This gives deeper effective receptive fields
# without adding more parameters.

class RCNN_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=512):
        super(RCNN_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                           # → 64×16×50

            # GRCL with 5 recurrent iterations (each iteration refines features
            # using both the constant input projection and the recurrent state)
            GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, 2),                                           # → 64×8×25

            GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),                              # → 128×4×26

            GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),                              # → 256×2×27

            # Final 2×1 conv: H 2→1
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True)         # → 512×1×26
        )

    def forward(self, input):
        return self.ConvNet(input)


# ─────────────────────────────────────────────────────────────────────────────
# GRCL — GATED RECURRENT CONVOLUTIONAL LAYER
# ─────────────────────────────────────────────────────────────────────────────
# One GRCL block applies two projections of the input (u) and then refines
# the hidden state x for num_iteration steps using those fixed projections
# plus learnable recurrent projections of x.
#
# The gate G(t) controls how much of the recurrent refinement to apply at
# each step (analogous to a GRU gate, but in 2-D feature space).
#
# Notation (from the paper):
#   u       = input feature map (constant across iterations)
#   x(t)    = recurrent hidden feature map at step t
#   wgf_u   = 1×1 conv of u  for the gate
#   wgr_x   = 1×1 conv of x  for the gate
#   wf_u    = k×k conv of u  for the state
#   wr_x    = k×k conv of x  for the state

class GRCL(nn.Module):

    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        # 1×1 convolutions project input u into the gate and state paths
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        # 1×1 conv of current hidden x for the gate path
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        # k×k conv of u for the state update
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        # k×k conv of current hidden x for the state update
        self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

        # BN for the initial state (x at t=0 from u only)
        self.BN_x_init = nn.BatchNorm2d(output_channel)

        self.num_iteration = num_iteration
        # Create num_iteration independent GRCL_unit instances (each has its own BN)
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, input):
        # Compute the input projections once — they are constant across all iterations
        wgf_u = self.wgf_u(input)   # gate projection of u
        wf_u  = self.wf_u(input)    # state projection of u

        # Initial hidden state x(0): just the state projection of u, no recurrence yet
        x = F.relu(self.BN_x_init(wf_u))

        # Iteratively refine x using the recurrent connections
        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCL_unit(nn.Module):
    # One recurrent step of GRCL:
    #   G  = sigmoid(BN(wgf_u) + BN(wgr_x))          — gate in [0,1]
    #   x  = relu(BN(wf_u) + BN(BN(wr_x) * G))       — gated state update
    #
    # G close to 1 → the recurrent update (wr_x) passes through fully.
    # G close to 0 → the hidden state stays close to the input projection (wf_u).

    def __init__(self, output_channel):
        super(GRCL_unit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output_channel)   # normalise gate input from u
        self.BN_grx = nn.BatchNorm2d(output_channel)   # normalise gate input from x
        self.BN_fu  = nn.BatchNorm2d(output_channel)   # normalise state input from u
        self.BN_rx  = nn.BatchNorm2d(output_channel)   # normalise state input from x
        self.BN_Gx  = nn.BatchNorm2d(output_channel)   # normalise gated recurrent term

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        # Gate: how much recurrent information to let through
        G = F.sigmoid(self.BN_gfu(wgf_u) + self.BN_grx(wgr_x))

        # State update: blend input projection with gated recurrent projection
        x = F.relu(self.BN_fu(wf_u) + self.BN_Gx(self.BN_rx(wr_x) * G))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# RESNET FEATURE EXTRACTOR  (same architecture as easyocr/model/modules.py)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet_FeatureExtractor(nn.Module):
    # Custom ResNet with layers=[1,2,5,3] BasicBlocks.
    # Identical to the generation1 inference backbone — see easyocr/model/modules.py
    # for the full annotated version.

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)


# ─────────────────────────────────────────────────────────────────────────────
# BASIC RESIDUAL BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    # Standard residual block:  conv→BN→ReLU→conv→BN + skip → ReLU
    # The skip connection lets gradients flow back unchanged, solving vanishing
    # gradient problems in deep stacks.
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample    # 1×1 conv to match channel dims for skip
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)   # bias=False: BN provides the bias

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)   # match channels/size for addition

        out += residual     # skip connection
        out = self.relu(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM RESNET  (text-recognition variant)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet(nn.Module):
    # 4-stage ResNet that collapses image height to 1 while preserving width.
    # Channel schedule with output_channel=512: 1→32→64→128→256→512→512
    # See easyocr/model/modules.py for the full spatial size walkthrough.

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2),
                                     output_channel, output_channel]   # [128,256,512,512]
        self.inplanes = int(output_channel / 8)   # 64

        # ── Stem: two 3×3 convs ───────────────────────────────────────────────
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)   # 1→32
        self.bn0_1   = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)   # 32→64
        self.bn0_2   = nn.BatchNorm2d(self.inplanes)
        self.relu    = nn.ReLU(inplace=True)

        # ── Stage 1: 1 BasicBlock, 128ch, preceded by 2×2 pool ───────────────
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # H/2, W/2
        self.layer1   = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1    = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0],
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(self.output_channel_block[0])

        # ── Stage 2: 2 BasicBlocks, 256ch, preceded by 2×2 pool ─────────────
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # H/2, W/2
        self.layer2   = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2    = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1],
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(self.output_channel_block[1])

        # ── Stage 3: 5 BasicBlocks, 512ch, preceded by asymmetric pool ───────
        # Asymmetric: stride (2,1) halves H but leaves W intact
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3   = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3    = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2],
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3      = nn.BatchNorm2d(self.output_channel_block[2])

        # ── Stage 4: 3 BasicBlocks, 512ch, then two final convs ──────────────
        self.layer4   = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        # conv4_1: kernel (2,2), stride (2,1) → H/2, W ~same
        self.conv4_1  = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                  kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1    = nn.BatchNorm2d(self.output_channel_block[3])
        # conv4_2: kernel (2,2), stride 1, no padding → H: 2→1 exactly
        self.conv4_2  = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                  kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2    = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        # First block may need a 1×1 downsample conv to align channel counts for the skip
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn0_1(self.conv0_1(x)))     # stem conv 1
        x = self.relu(self.bn0_2(self.conv0_2(x)))     # stem conv 2
        # [B, 64, 32, W]

        x = self.maxpool1(x)                           # [B, 64, 16, W/2]
        x = self.relu(self.bn1(self.conv1(self.layer1(x))))

        x = self.maxpool2(x)                           # [B, 128, 8, W/4]
        x = self.relu(self.bn2(self.conv2(self.layer2(x))))

        x = self.maxpool3(x)                           # [B, 256, 4, ~W/4]  asymmetric
        x = self.relu(self.bn3(self.conv3(self.layer3(x))))

        x = self.layer4(x)
        x = self.relu(self.bn4_1(self.conv4_1(x)))     # [B, 512, 2, ~W/4]
        x = self.relu(self.bn4_2(self.conv4_2(x)))     # [B, 512, 1, ~W/4]

        return x
