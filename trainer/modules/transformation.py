import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# TPS SPATIAL TRANSFORMER NETWORK  (text rectification)
# ─────────────────────────────────────────────────────────────────────────────
#
# Problem: curved, rotated, or perspectively-distorted text is much harder for
# the CNN to recognise than straight, horizontal text.
#
# Solution: learn to warp the input image into a "canonical" upright rectangle
# BEFORE the CNN sees it.  This is called Transformation stage 0 in model.py.
#
# How it works — three steps:
#
#   Step 1: LocalizationNetwork (a small CNN)
#     Takes the distorted input image and predicts F control-point positions C'
#     in the output (rectified) image space.  F is typically 20 (10 on top edge,
#     10 on bottom edge).
#
#   Step 2: GridGenerator
#     Uses Thin-Plate Spline (TPS) interpolation to compute, for every pixel
#     in the output grid, where it should be sampled from in the input image.
#     TPS is a smooth interpolation: given F known (source → target) point pairs,
#     it finds the smoothest possible warp that satisfies those constraints.
#
#   Step 3: F.grid_sample
#     Performs the actual image warp using bilinear interpolation at each
#     sampling location.  Differentiable → gradients flow back through the warp
#     into LocalizationNetwork, so the whole thing is trained end-to-end.
#
# Reference: RARE paper (Shi et al., CVPR 2016)

class TPS_SpatialTransformerNetwork(nn.Module):

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        # F:            number of fiducial (control) points
        # I_size:       (H, W) of the input image
        # I_r_size:     (H, W) of the rectified output image (same as I_size here)
        # I_channel_num: 1 for grayscale, 3 for RGB
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        # LocalizationNetwork predicts the F target control-point positions
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        # GridGenerator converts those positions into a full sampling grid
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        # batch_I: [B, C, H, W]  — distorted input images

        # Step 1: predict F control-point positions  [B, F, 2]
        batch_C_prime = self.LocalizationNetwork(batch_I)

        # Step 2: build sampling grid  [B, H_r, W_r, 2]
        #   Each (x, y) in the grid says: "to fill this output pixel, sample
        #   the input image at this (x, y) coordinate."
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # [B, H_r*W_r, 2]
        build_P_prime_reshape = build_P_prime.reshape(
            [build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])  # [B, H_r, W_r, 2]

        # Step 3: warp the input image using bilinear sampling
        # padding_mode='border': out-of-bounds samples take the nearest border value
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')
        # batch_I_r: [B, C, H_r, W_r]  — rectified image, same size as input

        return batch_I_r


# ─────────────────────────────────────────────────────────────────────────────
# LOCALIZATION NETWORK
# ─────────────────────────────────────────────────────────────────────────────
# A small CNN + two FC layers that maps an input image to F control-point (x,y)
# coordinates.  The key challenge is that the network must learn to find the
# top and bottom text boundaries from visual evidence alone.

class LocalizationNetwork(nn.Module):

    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        # Lightweight CNN: 4 conv+BN+ReLU blocks with MaxPool, ending in global avg pool.
        # Final output: [B, 512] — a compact representation of the whole image.
        self.conv = nn.Sequential(
            nn.Conv2d(self.I_channel_num, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),     # H/2, W/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),     # H/4, W/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),     # H/8, W/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1) # → [B, 512, 1, 1]  global spatial pooling
        )

        # Two FC layers: 512 → 256 → F*2 (F control points, each with x and y)
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # ── Smart initialisation for fc2 ──────────────────────────────────────
        # Start with zero weights so the prediction is purely from the bias.
        self.localization_fc2.weight.data.fill_(0)

        # Initialise the bias to evenly-spaced control points: F/2 on the top edge
        # and F/2 on the bottom edge of a [-1, 1] normalised coordinate system.
        # This gives the network a useful starting point — without this, early
        # training would predict random, unusable control-point positions.
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))           # x: -1 to +1
        ctrl_pts_y_top    = np.linspace(0.0, -1.0, num=int(F / 2)) # y: top row   (0 to -1)
        ctrl_pts_y_bottom = np.linspace(1.0,  0.0, num=int(F / 2)) # y: bottom row(1 to 0)
        ctrl_pts_top    = np.stack([ctrl_pts_x, ctrl_pts_y_top],    axis=1)  # [F/2, 2]
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)  # [F/2, 2]
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)  # [F, 2]
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        # batch_I: [B, C, H, W]
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)          # [B, 512]
        batch_C_prime = self.localization_fc2(
            self.localization_fc1(features)).view(batch_size, self.F, 2)
        # batch_C_prime: [B, F, 2]  — predicted control point (x,y) in [-1,1] space
        return batch_C_prime


# ─────────────────────────────────────────────────────────────────────────────
# GRID GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
# Computes the sampling grid using Thin-Plate Spline (TPS) interpolation.
#
# TPS interpolation finds the smoothest warp T such that:
#   T(C[i]) = C_prime[i]  for i = 0..F-1
# where C are the fixed canonical control-point positions and C_prime are the
# predicted positions from LocalizationNetwork.
#
# The math:
#   T = [1, x, y, φ(r_0), φ(r_1), ..., φ(r_{F-1})] × w
# where φ(r) = r² log(r) is the TPS radial basis function and w are solved by
# a linear system involving inv_delta_C.
#
# inv_delta_C and P_hat are constants that depend only on the canonical grid C
# and the output pixel positions P, so they are pre-computed once in __init__
# and registered as buffers (they move to GPU with the model but are not trained).

class GridGenerator(nn.Module):

    def __init__(self, F, I_r_size):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        # Fixed canonical control-point positions C: [F, 2]
        # F/2 points evenly spaced on the top edge, F/2 on the bottom edge.
        self.C = self._build_C(self.F)
        # Regular pixel grid of the output image: [H_r*W_r, 2]
        self.P = self._build_P(self.I_r_width, self.I_r_height)

        # Pre-compute and register as model buffers (not parameters — not trained):
        #   inv_delta_C: [F+3, F+3]  inverse of the TPS coefficient matrix
        #   P_hat:       [H*W, F+3]  TPS basis functions evaluated at every output pixel
        self.register_buffer("inv_delta_C",
            torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())
        self.register_buffer("P_hat",
            torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())

    def _build_C(self, F):
        # Canonical (reference) control points: F/2 on the top edge (y=-1),
        # F/2 on the bottom edge (y=+1), x evenly spread in [-1, 1].
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_top    = np.stack([ctrl_pts_x, -1 * np.ones(int(F / 2))], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x,  1 * np.ones(int(F / 2))], axis=1)
        return np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)  # [F, 2]

    def _build_inv_delta_C(self, F, C):
        # Builds and inverts the TPS coefficient matrix delta_C.
        # The matrix encodes the distances between all pairs of control points
        # through the TPS radial basis function φ(r) = r² log(r).
        # Its inverse is used to solve for the TPS weights given any C_prime.
        hat_C = np.zeros((F, F), dtype=float)
        for i in range(F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])    # Euclidean distance
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)       # φ(r) = r² log(r)

        # delta_C is the full (F+3)×(F+3) TPS system matrix:
        # [  1  C  hat_C ]   F rows
        # [  0  0  C^T   ]   2 rows  (affine constraints)
        # [  0  1  1...1 ]   1 row
        delta_C = np.concatenate([
            np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),          # F × (F+3)
            np.concatenate([np.zeros((2, 3)), np.transpose(C)],  axis=1), # 2 × (F+3)
            np.concatenate([np.zeros((1, 3)), np.ones((1, F))],  axis=1)  # 1 × (F+3)
        ], axis=0)
        return np.linalg.inv(delta_C)   # [F+3, F+3]

    def _build_P(self, I_r_width, I_r_height):
        # Regular grid of all output pixel positions, normalised to [-1, 1].
        # This is the set of points P we want to find source coordinates for.
        I_r_grid_x = (np.arange(-I_r_width,  I_r_width,  2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        P = np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)  # [H, W, 2]
        return P.reshape([-1, 2])   # [H*W, 2]

    def _build_P_hat(self, F, C, P):
        # Evaluates the TPS basis functions at every output pixel position P.
        # For each pixel p and each control point c_j:
        #   φ(||p - c_j||) = ||p - c_j||² log(||p - c_j||)
        # P_hat = [1, p_x, p_y, φ(||p-c_0||), ..., φ(||p-c_{F-1}||)] for each p
        n = P.shape[0]   # H*W
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # [n, F, 2]
        C_tile = np.expand_dims(C, axis=0)                       # [1, F, 2]
        P_diff = P_tile - C_tile                                 # [n, F, 2]
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2)         # [n, F]  distances
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # [n, F]  φ(r)
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)  # [n, F+3]
        return P_hat

    def build_P_prime(self, batch_C_prime):
        # Given predicted control-point positions C_prime [B, F, 2],
        # compute the source coordinates for every output pixel.
        #
        # Step 1: solve for TPS weights T using the pre-inverted matrix:
        #   T = inv_delta_C × [C_prime; zeros]   → [B, F+3, 2]
        # Step 2: evaluate the warp at all output pixels:
        #   P_prime = P_hat × T                  → [B, H*W, 2]
        #
        # P_prime[b, i] = (x_src, y_src) for output pixel i in sample b.

        batch_size = batch_C_prime.size(0)

        # Expand precomputed constants to match the batch size
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)  # [B, F+3, F+3]
        batch_P_hat       = self.P_hat.repeat(batch_size, 1, 1)        # [B, H*W, F+3]

        # Append 3 zero rows to C_prime to satisfy the affine constraints of TPS
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime,
             torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)   # [B, F+3, 2]

        # Solve TPS: T = inv_delta_C × C_prime_with_zeros
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # [B, F+3, 2]

        # Evaluate warp at all output pixel locations
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # [B, H*W, 2]
        return batch_P_prime   # source coordinates for each output pixel
