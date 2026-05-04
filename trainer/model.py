import torch.nn as nn
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

# ─────────────────────────────────────────────────────────────────────────────
# TRAINER MODEL  (more flexible than the inference model in easyocr/model/)
# ─────────────────────────────────────────────────────────────────────────────
#
# This Model class is configured entirely by `opt` flags, making it easy to
# swap each of the four stages independently:
#
#  Stage 0 — Transformation (opt.Transformation):
#    'None' : skip, pass image straight to CNN
#    'TPS'  : Thin-Plate Spline spatial transformer — rectifies curved/tilted text
#
#  Stage 1 — Feature Extraction (opt.FeatureExtraction):
#    'VGG'    : lightweight VGG-style CNN
#    'RCNN'   : residual CNN (similar to VGG but with skip connections)
#    'ResNet' : deeper custom ResNet (same as generation1 in inference)
#
#  Stage 2 — Sequence Modeling (opt.SequenceModeling):
#    'None'   : skip, pass CNN features directly to prediction head
#    'BiLSTM' : 2 stacked bidirectional LSTMs
#
#  Stage 3 — Prediction (opt.Prediction):
#    'CTC'  : linear layer + CTC loss (no explicit alignment needed)
#    'Attn' : attention-based seq2seq decoder (slower, sometimes more accurate)

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        # Keep a dict of which variant is active — used in forward() to branch
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        # ── Stage 0: Transformation ───────────────────────────────────────────
        if opt.Transformation == 'TPS':
            # TPS_SpatialTransformerNetwork:
            #   1. A small CNN (LocalizationNetwork) predicts F control-point locations
            #      from the input image.
            #   2. GridGenerator uses those points to compute a sampling grid via
            #      Thin-Plate Spline interpolation.
            #   3. F.grid_sample warps the image to a canonical upright rectangle.
            # F = opt.num_fiducial: number of control points (typically 20).
            # Helps with curved, slanted, or perspectively distorted text.
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        # ── Stage 1: Feature Extraction ───────────────────────────────────────
        # All three CNNs output shape [B, output_channel, H', W'] where H'≈1.
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel

        # Collapses the residual height dimension so every backbone produces
        # exactly [B, W', output_channel] regardless of architecture.
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # ── Stage 2: Sequence Modeling ─────────────────────────────────────────
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            # If no LSTM, the prediction head reads CNN features directly.
            self.SequenceModeling_output = self.FeatureExtraction_output

        # ── Stage 3: Prediction ───────────────────────────────────────────────
        if opt.Prediction == 'CTC':
            # Simple linear layer: [B, T, hidden] → [B, T, num_class]
            # Loss: CTCLoss (handles alignment between T columns and variable-length text)
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            # Attention decoder: at each decoding step it attends over all T CNN
            # columns and emits one character, continuing until [s] (end-of-seq).
            # Requires teacher-forcing during training (see train.py).
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        # `text` usage depends on the prediction head:
        #   CTC:  not used (passed as dummy)
        #   Attn: ground-truth token sequence for teacher-forcing (train) or
        #         start token for autoregressive decoding (eval)
        # `is_train`: controls teacher-forcing vs. greedy decoding in Attention

        # ── Stage 0 ──────────────────────────────────────────────────────────
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
        # input still [B, C, H, W], but now geometrically rectified

        # ── Stage 1 ──────────────────────────────────────────────────────────
        visual_feature = self.FeatureExtraction(input)
        # [B, output_channel, H', W']

        # Rearrange into sequence: [B, C, H', W'] → [B, W', C, 1] → [B, W', C]
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        # [B, W', output_channel]

        # ── Stage 2 ──────────────────────────────────────────────────────────
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature   # bypass LSTM
        # [B, W', hidden_size]

        # ── Stage 3 ──────────────────────────────────────────────────────────
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
            # [B, W', num_class]  — raw logits, log_softmax applied in train.py
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                         batch_max_length=self.opt.batch_max_length)
            # [B, max_length, num_class]  — one score distribution per decoded step

        return prediction
