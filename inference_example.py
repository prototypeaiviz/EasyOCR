"""
EasyOCR Recognition Model — Step-by-Step Inference Example
============================================================
Run:
    python inference_example.py path/to/your/image.png

The image should be a crop of a single word or short line of text
(the kind of image the DETECTOR would normally hand to the recognizer).
If you give it a full scene image it will still run but accuracy will be low.

Set LANGUAGE below to match what's in your image.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import math

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "your_image.png"

# Change to whatever language(s) your image contains.
# Common options: ['en'], ['en','ar'], ['ch_sim','en'], etc.
LANGUAGE = ['en']

# Decoding strategy: 'greedy' (fastest), 'beamsearch', 'wordbeamsearch'
DECODER = 'greedy'
BEAM_WIDTH = 5

# Target height the model expects (32 for most models, 64 for some)
IMG_H = 32
# Maximum width (images wider than this are clamped)
IMG_W = 320

# ── 1. LOAD MODEL via EasyOCR Reader ─────────────────────────────────────────
# Reader handles:
#   - Downloading weights if not already present
#   - Building CTCLabelConverter (character ↔ index mapping)
#   - Loading the correct architecture (gen1 ResNet or gen2 VGG)
#   - Moving to GPU if available, or applying int8 quantization on CPU

print("\n" + "="*70)
print("STEP 1 — Loading model via easyocr.Reader")
print("="*70)

import easyocr
reader = easyocr.Reader(LANGUAGE, gpu=False, verbose=False)

# After Reader.__init__:
#   reader.recognizer  → the nn.Module (Model class)
#   reader.converter   → CTCLabelConverter instance
#   reader.character   → list of characters the model knows

recognizer = reader.recognizer
converter  = reader.converter
device     = reader.device

print(f"Device       : {device}")
print(f"Vocabulary   : {len(converter.character)} tokens  (index 0 = '[blank]', 1..N = real chars)")
print(f"First 20 chars: {converter.character[:20]}")

# ── 2. LOAD & INSPECT THE RAW IMAGE ──────────────────────────────────────────

print("\n" + "="*70)
print("STEP 2 — Loading raw image")
print("="*70)

raw_img = Image.open(IMAGE_PATH).convert('L')   # force grayscale ('L' mode)
print(f"PIL image size (W×H): {raw_img.size}")
print(f"Mode: {raw_img.mode}")

# ── 3. PREPROCESSING — Resize + Aspect-ratio pad ─────────────────────────────
# Mirrors what AlignCollate + NormalizePAD do inside get_text().

print("\n" + "="*70)
print("STEP 3 — Preprocessing: resize to H=32, pad to fixed width")
print("="*70)

w_orig, h_orig = raw_img.size
ratio = w_orig / float(h_orig)
resized_w = min(math.ceil(IMG_H * ratio), IMG_W)   # preserve aspect ratio, cap at IMG_W

resized_img = raw_img.resize((resized_w, IMG_H), Image.BICUBIC)
print(f"After resize (W×H): {resized_img.size}  (aspect-ratio preserved, H fixed to {IMG_H})")

# NormalizePAD: convert to tensor, normalise to [-1,1], right-pad to IMG_W
import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()

img_tensor = to_tensor(resized_img)          # [1, H, W], float32 in [0,1]
img_tensor = img_tensor.sub_(0.5).div_(0.5)  # [1, H, W], float32 in [-1,1]
print(f"After ToTensor + normalise: {tuple(img_tensor.shape)}  range [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

# Pad the right side with the last column (avoids a hard black edge artefact)
C, H, W = img_tensor.shape
pad_img = torch.FloatTensor(1, H, IMG_W).fill_(0)
pad_img[:, :, :W] = img_tensor
if IMG_W > W:
    pad_img[:, :, W:] = img_tensor[:, :, W-1].unsqueeze(2).expand(C, H, IMG_W - W)

print(f"After right-pad to IMG_W={IMG_W}: {tuple(pad_img.shape)}")

# Add batch dimension: [1, H, IMG_W] → [1, 1, H, IMG_W]
batch = pad_img.unsqueeze(0).to(device)
print(f"Batch tensor:                   {tuple(batch.shape)}  ← [B, C, H, W]")

# ── 4. FORWARD PASS — CNN Feature Extraction ─────────────────────────────────

print("\n" + "="*70)
print("STEP 4 — CNN Feature Extraction")
print("="*70)

recognizer.eval()
with torch.no_grad():

    # Pull the underlying model out of DataParallel if on GPU
    m = recognizer.module if hasattr(recognizer, 'module') else recognizer

    # --- Stage 1: CNN ---------------------------------------------------------
    # For gen1 (ResNet): ResNet collapses H from 32 → 1 via asymmetric pooling
    # For gen2 (VGG):    VGG does the same with plain Conv+Pool blocks
    cnn_features = m.FeatureExtraction(batch)
    print(f"CNN output:   {tuple(cnn_features.shape)}  ← [B, C, H', W']")
    print(f"  H' = {cnn_features.shape[2]}  (should be ~1 after all the pooling)")
    print(f"  W' = {cnn_features.shape[3]}  (roughly = image_width / 4, preserving char positions)")

    # ── 5. RESHAPE BRIDGE: 2D feature map → 1D sequence ─────────────────────

    print("\n" + "="*70)
    print("STEP 5 — Reshape: [B,C,H',W'] → [B,W',C]  (image → sequence)")
    print("="*70)

    # permute(0,3,1,2): [B, C, H', W'] → [B, W', C, H']   (W' becomes the time axis)
    seq = m.AdaptiveAvgPool(cnn_features.permute(0, 3, 1, 2))
    print(f"After permute + AdaptiveAvgPool: {tuple(seq.shape)}  ← [B, W', C, 1]")

    seq = seq.squeeze(3)
    print(f"After squeeze (remove H'=1):     {tuple(seq.shape)}  ← [B, W', C]")
    print(f"  Each of the {seq.shape[1]} time steps is a {seq.shape[2]}-dim feature vector")
    print(f"  describing one vertical column of the image")

    # ── 6. BILSTM SEQUENCE MODELING ───────────────────────────────────────────

    print("\n" + "="*70)
    print("STEP 6 — BiLSTM: add left+right context across the sequence")
    print("="*70)

    context = m.SequenceModeling(seq)
    print(f"BiLSTM output: {tuple(context.shape)}  ← [B, W', hidden_size]")
    print(f"  At each time step the LSTM has now 'seen' all columns (fwd+bwd pass)")

    # ── 7. LINEAR PREDICTION HEAD ─────────────────────────────────────────────

    print("\n" + "="*70)
    print("STEP 7 — Linear head: hidden_size → num_class (raw logits)")
    print("="*70)

    # text_for_pred is a dummy (not used by CTC models, only by Attention models)
    batch_max_length = int(IMG_W / 10)
    text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(device)

    logits = m.Prediction(context.contiguous())
    print(f"Raw logits:    {tuple(logits.shape)}  ← [B, T, num_class]")
    print(f"  T = {logits.shape[1]} time steps  ×  {logits.shape[2]} classes (incl. blank at index 0)")

    # ── 8. SOFTMAX + RE-NORMALISE ─────────────────────────────────────────────

    print("\n" + "="*70)
    print("STEP 8 — Softmax + re-normalise (zero out ignored tokens)")
    print("="*70)

    preds_prob = F.softmax(logits, dim=2)        # [B, T, C], each row sums to 1
    preds_prob_np = preds_prob.cpu().numpy()

    print(f"After softmax: {preds_prob_np.shape}  (values in [0,1], sum per time step = 1.0)")
    print(f"  Max prob at step 0: {preds_prob_np[0,0].max():.4f}  "
          f"(class {preds_prob_np[0,0].argmax()} = '{converter.character[preds_prob_np[0,0].argmax()]}')")

    # ignore_idx: blank (0) + any separator tokens
    ignore_idx = converter.ignore_idx
    preds_prob_np[:, :, ignore_idx] = 0.0       # zero out — they can't win argmax

    # re-normalise so remaining probs still sum to 1
    row_sums = preds_prob_np.sum(axis=2, keepdims=True)
    preds_prob_np = preds_prob_np / np.maximum(row_sums, 1e-8)
    print(f"After zeroing ignored tokens and re-normalising: same shape {preds_prob_np.shape}")

    # ── 9. DECODING ───────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print(f"STEP 9 — CTC Decoding  (mode: {DECODER})")
    print("="*70)

    preds_prob_t = torch.from_numpy(preds_prob_np).float().to(device)
    T = logits.shape[1]
    preds_size = torch.IntTensor([T])

    if DECODER == 'greedy':
        # At each time step take argmax → index of most probable class
        _, preds_index = preds_prob_t.max(2)           # [B, T]
        print(f"Argmax indices (raw, before CTC collapse):")
        print(f"  {preds_index[0].tolist()}")
        print(f"  Corresponding chars: {[converter.character[i] for i in preds_index[0].tolist()]}")

        preds_index_flat = preds_index.view(-1)         # flatten for batch decode
        preds_str = converter.decode_greedy(
            preds_index_flat.data.cpu().numpy(), preds_size.data)

    elif DECODER == 'beamsearch':
        k = preds_prob_np
        preds_str = converter.decode_beamsearch(k, beamWidth=BEAM_WIDTH)

    elif DECODER == 'wordbeamsearch':
        k = preds_prob_np
        preds_str = converter.decode_wordbeamsearch(k, beamWidth=BEAM_WIDTH)

    print(f"\nDecoded text: '{preds_str[0]}'")

    # ── 10. CONFIDENCE SCORE ──────────────────────────────────────────────────

    print("\n" + "="*70)
    print("STEP 10 — Confidence score")
    print("="*70)

    values  = preds_prob_np.max(axis=2)           # [B, T] — winning prob at each step
    indices = preds_prob_np.argmax(axis=2)        # [B, T] — winning class at each step

    # Keep only non-blank time steps (index != 0 after re-normalisation argmax)
    # These correspond to the character positions the model committed to.
    winning_classes = indices[0]                  # [T] for sample 0
    winning_probs   = values[0]                   # [T]

    non_blank_mask = winning_classes != 0
    char_probs = winning_probs[non_blank_mask]

    print(f"Non-blank time steps: {non_blank_mask.sum()} out of {T}")
    print(f"Per-character probs:  {[f'{p:.3f}' for p in char_probs]}")

    if len(char_probs) == 0:
        confidence = 0.0
    else:
        # custom_mean from recognition.py:  prod(probs) ^ (2 / sqrt(n))
        # Modified geometric mean that slightly penalises longer words.
        confidence = float(char_probs.prod() ** (2.0 / np.sqrt(len(char_probs))))

    print(f"\nConfidence score: {confidence:.4f}  (range [0,1], >0.5 is typically reliable)")

    # ── FINAL RESULT ──────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"  Text       : '{preds_str[0]}'")
    print(f"  Confidence : {confidence:.4f}")
    print("="*70 + "\n")
