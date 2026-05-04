import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# BIDIRECTIONAL LSTM  (sequence modeling stage)
# ─────────────────────────────────────────────────────────────────────────────
#
# Role in the pipeline:
#   The CNN feature extractor produces a sequence of column feature vectors,
#   one per horizontal strip of the image.  Each vector describes only the
#   local visual content of that strip.
#
#   The BiLSTM reads this sequence and adds temporal context: at each position t,
#   the output vector summarises not just what's at column t, but also what
#   came before (forward LSTM) and what comes after (backward LSTM).
#
#   This is crucial for OCR because many characters are visually ambiguous in
#   isolation ('l' vs '1' vs 'I', 'rn' vs 'm') but clear in context.
#
# Two layers are stacked:
#   Layer 1:  CNN_output_channel (512) → hidden_size (256)
#             Compresses CNN features into a compact context representation.
#   Layer 2:  hidden_size (256) → hidden_size (256)
#             Further refines temporal reasoning with a deeper representation.
#
# Bidirectional mechanics:
#   nn.LSTM(..., bidirectional=True) runs two independent LSTMs:
#     forward:  processes columns left-to-right  → hidden_fwd [B, T, H]
#     backward: processes columns right-to-left  → hidden_bwd [B, T, H]
#   Their outputs are concatenated → [B, T, 2H].
#   A Linear layer then projects [B, T, 2H] → [B, T, output_size]
#   so the output dimension matches the input dimension of the next layer.

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        # bidirectional=True: runs forward and backward LSTMs in parallel.
        # batch_first=True: tensor shape is [B, T, features] throughout
        #   (default PyTorch LSTM is [T, B, features] which is less intuitive).
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)

        # Project the concatenated [forward || backward] hidden state back to output_size.
        # hidden_size * 2: forward and backward hidden states concatenated.
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # input:  [B, T, input_size]
        # output: [B, T, output_size]

        try:
            # flatten_parameters() makes LSTM weight matrices contiguous in GPU memory.
            # This is required when using nn.DataParallel (multi-GPU), because the
            # parameter tensors may be scattered across non-contiguous memory regions.
            self.rnn.flatten_parameters()
        except:
            # Dynamic quantisation (CPU inference) replaces the LSTM internals with
            # quantised kernels, so flatten_parameters() is no longer valid.
            pass

        recurrent, _ = self.rnn(input)
        # recurrent: [B, T, 2*hidden_size]
        # The second return value is (h_n, c_n) — final hidden and cell states
        # across the sequence.  We discard them: we need all T outputs, not just
        # the final state.

        output = self.linear(recurrent)
        # output: [B, T, output_size]
        return output
