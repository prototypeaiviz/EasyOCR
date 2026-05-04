import torch.nn as nn
from .modules import ResNet_FeatureExtractor, BidirectionalLSTM

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION 1 RECOGNITION MODEL  (ResNet backbone)
# ─────────────────────────────────────────────────────────────────────────────
#
# Architecture overview (CRNN — Convolutional Recurrent Neural Network):
#
#   Input image  [B, 1, H, W]    (B=batch, 1=grayscale, H=32, W=variable/padded)
#        │
#   ┌────▼──────────────────────────────────┐
#   │  ResNet_FeatureExtractor (CNN)         │  extracts local visual features
#   │  Output: [B, 512, H', W']              │  H' ≈ 1-2 rows, W' ≈ W/4
#   └────────────────────────────────────────┘
#        │  permute(0,3,1,2) → [B, W', 512, H']
#        │  AdaptiveAvgPool2d((None,1)) → [B, W', 512, 1]   collapse height
#        │  squeeze(3) → [B, W', 512]
#        │
#        │  Now the tensor is a sequence of W' feature vectors, one per
#        │  horizontal "column slice" of the image — each vector describes
#        │  what the CNN saw in that vertical strip.
#        │
#   ┌────▼──────────────────────────────────┐
#   │  BiLSTM × 2 (Sequence Modeling)        │  captures context across columns
#   │  Output: [B, W', hidden_size]          │
#   └────────────────────────────────────────┘
#        │
#   ┌────▼──────────────────────────────────┐
#   │  Linear(hidden_size → num_class)       │  score each character per time step
#   │  Output: [B, W', num_class]            │  raw logits (no softmax here)
#   └────────────────────────────────────────┘
#        │
#   Decoded by CTC (Connectionist Temporal Classification) in recognition.py
#   → final predicted string (variable length)
#
# Why CTC?  The model outputs one set of scores per column (W' scores total)
# but the actual word may have fewer characters than W'.  CTC handles the
# alignment automatically — it learns which columns correspond to which
# characters and which are "blank" (no character here).

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        # input_channel:  1 for grayscale images
        # output_channel: number of feature maps out of the CNN (typically 512)
        # hidden_size:    LSTM hidden dimension (typically 256)
        # num_class:      size of the character vocabulary + 1 (CTC blank)
        super(Model, self).__init__()

        # ── Stage 1: Feature Extraction ──────────────────────────────────────
        # Custom ResNet that aggressively reduces height (H→1) while preserving
        # width (W stays roughly W/4), turning a 2-D image into a 1-D sequence.
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # == 512 typically

        # CollapseHeight: pools over the remaining height dimension.
        # (None, 1) means: keep the width dimension as-is, pool height down to 1.
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # ── Stage 2: Sequence Modeling ────────────────────────────────────────
        # Two stacked Bidirectional LSTMs.
        # The first LSTM reads the 512-d CNN features and produces hidden_size-d vectors.
        # The second LSTM refines those vectors with more temporal context.
        # "Bidirectional" means each time step sees context from both left AND right,
        # which is important: recognising the middle of a word benefits from knowing
        # what came before AND what comes after.
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        # ── Stage 3: Prediction ───────────────────────────────────────────────
        # A simple linear layer that converts each hidden vector into a score
        # distribution over the character vocabulary (including blank).
        # No softmax here — it's applied later in recognizer_predict().
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        # `text` is a dummy tensor (zeros) passed in from recognizer_predict.
        # It is not used here; it exists because attention-based model variants
        # need it for teacher-forcing during training/decoding.

        # ── Stage 1: Feature extraction ──────────────────────────────────────
        visual_feature = self.FeatureExtraction(input)
        # visual_feature shape: [B, output_channel, H', W']
        # e.g. [8, 512, 1, 26] for a 32×100 input

        # permute(0,3,1,2): [B, C, H', W'] → [B, W', C, H']
        # Now dimension 1 is the width (time axis) and dimension 3 is height.
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        # After pool: [B, W', C, 1]

        visual_feature = visual_feature.squeeze(3)
        # squeeze dim 3: [B, W', C]  — ready to be treated as a sequence

        # ── Stage 2: Sequence modeling ────────────────────────────────────────
        # Input:  [B, W', 512]
        # Output: [B, W', hidden_size]
        contextual_feature = self.SequenceModeling(visual_feature)

        # ── Stage 3: Prediction ───────────────────────────────────────────────
        # Linear applied independently to each of the W' time steps.
        # Input:  [B, W', hidden_size]
        # Output: [B, W', num_class]  — raw logits per time step per class
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
