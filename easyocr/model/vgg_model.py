import torch.nn as nn
from .modules import VGG_FeatureExtractor, BidirectionalLSTM

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION 2 RECOGNITION MODEL  (VGG backbone)
# ─────────────────────────────────────────────────────────────────────────────
#
# Identical pipeline to generation1/model.py but uses a lighter VGG-style CNN
# instead of the custom ResNet.  VGG_FeatureExtractor uses plain Conv→ReLU→Pool
# blocks (no residual connections), which is faster but slightly less accurate
# for complex scripts.
#
# Full pipeline:
#
#   Input [B, 1, H=32, W]
#        │
#   VGG_FeatureExtractor  → [B, output_channel, H', W']
#        │  permute + AdaptiveAvgPool + squeeze → [B, W', output_channel]
#   BiLSTM × 2            → [B, W', hidden_size]
#   Linear                → [B, W', num_class]   (raw CTC logits)

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        # input_channel:  1 (grayscale)
        # output_channel: CNN feature map depth (e.g. 256)
        # hidden_size:    LSTM hidden size (e.g. 256)
        # num_class:      vocab size + 1 CTC blank token
        super(Model, self).__init__()

        # ── Stage 1: Feature Extraction ──────────────────────────────────────
        # VGG-style CNN: stacked Conv→ReLU blocks with MaxPool layers that
        # shrink H faster than W, eventually collapsing H to ~1.
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel

        # Pools away any residual height so every sample has exactly H'=1.
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # ── Stage 2: Sequence Modeling ────────────────────────────────────────
        # Two stacked BiLSTMs read the sequence of column features and add
        # left-right context, crucial for ambiguous characters like 'l', '1', 'I'.
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        # ── Stage 3: Prediction ───────────────────────────────────────────────
        # Maps each time-step's hidden vector to raw class scores.
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        # `text` is unused (exists for API compatibility with attention models).

        # ── Stage 1 ──────────────────────────────────────────────────────────
        visual_feature = self.FeatureExtraction(input)
        # [B, C, H', W']

        # Rearrange so width is the sequence axis:
        #   permute(0,3,1,2): [B, C, H', W'] → [B, W', C, H']
        #   AdaptiveAvgPool:  [B, W', C, H'] → [B, W', C, 1]
        #   squeeze:          [B, W', C, 1]  → [B, W', C]
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        contextual_feature = self.SequenceModeling(visual_feature)
        # [B, W', hidden_size]

        # ── Stage 3 ──────────────────────────────────────────────────────────
        prediction = self.Prediction(contextual_feature.contiguous())
        # [B, W', num_class]  — decoded by CTC in recognition.py

        return prediction
