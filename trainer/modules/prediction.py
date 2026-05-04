import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION-BASED SEQUENCE DECODER
# ─────────────────────────────────────────────────────────────────────────────
#
# Alternative to CTC prediction.  Instead of assigning one character per CNN
# column and letting CTC figure out the alignment, the attention decoder
# generates characters one at a time, attending over ALL columns at each step.
#
# At decoding step i:
#   1. AttentionCell computes an attention weight α over all T encoder columns.
#      α[j] = how much does column j contribute to predicting character i?
#   2. A context vector is formed as the weighted sum of encoder outputs.
#   3. The context vector + previous character token → LSTM → new hidden state.
#   4. Linear(hidden) → logits over vocab → softmax → predicted character i.
#
# Training mode (is_train=True):  teacher-forcing
#   The ground-truth previous character is fed at each step regardless of what
#   the model predicted.  This makes training stable and fast — the model sees
#   correct prefixes and learns to predict the next character.
#
# Eval mode (is_train=False):  autoregressive decoding
#   The model's own previous prediction is fed as input to the next step.
#   Slower (sequential), but gives realistic inference behaviour.
#
# Comparison with CTC:
#   CTC:  simpler, faster, scales better to long sequences, requires no explicit
#         start/end tokens.  Standard choice in EasyOCR.
#   Attn: more expressive, handles irregular character spacing better, but
#         slower and more sensitive to exposure bias (train/test mismatch).

class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        # AttentionCell handles one decoding step: attend + update hidden state
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # Final linear: hidden_size → num_classes (character scores)
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        # Convert a batch of character indices [B] to one-hot vectors [B, vocab].
        # Used because the decoder needs a discrete, fixed-size representation of
        # the previous character (not a floating-point embedding) as its input.
        input_char = input_char.unsqueeze(1)        # [B] → [B, 1]
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        # scatter_: one_hot[b, input_char[b]] = 1 for each sample b
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot                              # [B, vocab]

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        # batch_H: encoder (BiLSTM) output  [B, T, input_size]
        # text:    [B, max_length+1] integer token indices
        #          text[:, 0] = [GO] token (index 0) for all samples
        # Returns: [B, num_steps, num_classes]  — raw logits per decoding step

        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 to also predict the [s] end token

        # Pre-allocate output storage  [B, num_steps, hidden_size]
        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)

        # Initialise the decoder LSTM state with zeros
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),   # h_0
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))   # c_0

        if is_train:
            # ── Teacher-forcing (training) ────────────────────────────────────
            # At step i we feed the ground-truth token text[:, i] as input.
            # This means each step is conditioned on a correct prefix, making
            # gradients cleaner and training faster.
            for i in range(num_steps):
                # One-hot encode the ground-truth character for this step
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # Update hidden state by attending over encoder outputs + current char
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                # Store the hidden state for this step
                output_hiddens[:, i, :] = hidden[0]   # hidden[0] = h_t, hidden[1] = c_t

            # Project all hidden states to character scores in one batched call
            probs = self.generator(output_hiddens)    # [B, num_steps, num_classes]

        else:
            # ── Autoregressive decoding (eval/inference) ──────────────────────
            # Seed with [GO] token (index 0) for all samples in the batch.
            targets = torch.LongTensor(batch_size).fill_(0).to(device)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                # Generate the character score distribution for this step
                probs_step = self.generator(hidden[0])  # [B, num_classes]
                probs[:, i, :] = probs_step
                # Greedily pick the most probable character as input to the next step
                _, next_input = probs_step.max(1)       # [B]
                targets = next_input                    # feed own prediction forward

        return probs   # [B, num_steps, num_classes]  raw logits


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION CELL  (one decoding step)
# ─────────────────────────────────────────────────────────────────────────────
#
# Implements Bahdanau-style additive attention + an LSTMCell update:
#
#   1. Project encoder outputs H: [B, T, input_size] → [B, T, hidden_size]
#   2. Project previous decoder hidden h_{t-1}: [B, hidden_size] → [B, 1, hidden_size]
#      (unsqueeze so it broadcasts across T encoder positions)
#   3. Energy e: tanh(H_proj + h_proj) → Linear(1) → [B, T, 1]
#      Each e[b, j] measures how well decoder position t-1 aligns with encoder j.
#   4. α = softmax(e, dim=1) → attention weights [B, T, 1]
#      α[b, j] ∈ [0,1], sums to 1 across T.
#   5. context = bmm(α^T, H) → [B, 1, input_size] → squeeze → [B, input_size]
#      Weighted sum of encoder columns, focused on the most relevant part.
#   6. LSTMCell([context || char_onehot], prev_hidden) → new hidden state

class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        # Project encoder output into attention space (no bias: added in h2h)
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        # Project previous decoder hidden state into attention space
        self.h2h = nn.Linear(hidden_size, hidden_size)  # has bias for the sum
        # Scalar energy score per encoder position
        self.score = nn.Linear(hidden_size, 1, bias=False)
        # Decoder LSTM cell: input = context (input_size) + char one-hot (num_embeddings)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # ── Attention ─────────────────────────────────────────────────────────
        # Project all encoder positions to hidden_size
        batch_H_proj = self.i2h(batch_H)                    # [B, T, hidden_size]
        # Project previous hidden state and broadcast across T positions
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)  # [B, 1, hidden_size]

        # Additive energy: tanh( encoder_proj + hidden_proj ) for each position
        # Broadcasting makes prev_hidden_proj add to each of the T positions.
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # [B, T, 1]

        # Normalise to a probability distribution over encoder positions
        alpha = F.softmax(e, dim=1)                         # [B, T, 1]

        # Compute context vector: weighted sum of encoder outputs
        # alpha.permute(0,2,1): [B, 1, T]
        # batch_H: [B, T, input_size]
        # bmm result: [B, 1, input_size] → squeeze → [B, input_size]
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)

        # ── Decoder update ────────────────────────────────────────────────────
        # Concatenate context vector with current character embedding
        concat_context = torch.cat([context, char_onehots], 1)  # [B, input_size + num_embeddings]

        # LSTMCell: one recurrent step
        # prev_hidden = (h_{t-1}, c_{t-1}), cur_hidden = (h_t, c_t)
        cur_hidden = self.rnn(concat_context, prev_hidden)      # returns (h_t, c_t)

        return cur_hidden, alpha
