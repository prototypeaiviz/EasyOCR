# EasyOCR Recognition Model — Architecture Deep Dive

This document explains exactly how the EasyOCR text recognition system works at inference time: every component, every tensor transformation, and why each design choice was made.

---

## Table of Contents

1. [What the Recognition Model Does](#1-what-the-recognition-model-does)
2. [Two Model Generations](#2-two-model-generations)
3. [End-to-End Pipeline Overview](#3-end-to-end-pipeline-overview)
4. [Stage 0 — Image Preprocessing](#4-stage-0--image-preprocessing)
5. [Stage 1 — Feature Extraction (CNN)](#5-stage-1--feature-extraction-cnn)
   - [Generation 1: ResNet Backbone](#generation-1-resnet-backbone)
   - [Generation 2: VGG Backbone](#generation-2-vgg-backbone)
   - [The Key Design Insight: Asymmetric Pooling](#the-key-design-insight-asymmetric-pooling)
6. [The Reshape Bridge: CNN → Sequence](#6-the-reshape-bridge-cnn--sequence)
7. [Stage 2 — Sequence Modeling (BiLSTM)](#7-stage-2--sequence-modeling-bilstm)
8. [Stage 3 — Prediction Head (Linear)](#8-stage-3--prediction-head-linear)
9. [CTC Decoding](#9-ctc-decoding)
   - [What is CTC?](#what-is-ctc)
   - [Greedy Decoding](#greedy-decoding)
   - [Beam Search Decoding](#beam-search-decoding)
   - [Word Beam Search Decoding](#word-beam-search-decoding)
10. [Confidence Scoring](#10-confidence-scoring)
11. [Two-Pass Contrast Strategy](#11-two-pass-contrast-strategy)
12. [Model Loading and Quantization](#12-model-loading-and-quantization)
13. [Complete Tensor Shape Walkthrough](#13-complete-tensor-shape-walkthrough)
14. [Key Files Reference](#14-key-files-reference)

---

## 1. What the Recognition Model Does

The recognition model takes a **text crop** — a small rectangular image containing a single word or line of text — and outputs the **text string** and a **confidence score**.

```
Input:  numpy array (grayscale image crop, e.g. 32×87 pixels)
Output: ("Hello", 0.94)
```

This model is **not** responsible for finding where text is in the image — that is the detection model (CRAFT or DBNet). The recognition model only reads text that has already been located and cropped.

---

## 2. Two Model Generations

EasyOCR ships two recognition model architectures. Which one is used depends on the language:

| Generation | Backbone | Output channels | File |
|-----------|----------|-----------------|------|
| **gen1** | Custom ResNet | 512 | `easyocr/model/model.py` |
| **gen2** | VGG-style | 256 | `easyocr/model/vgg_model.py` |

Both use the **exact same pipeline** (CNN → BiLSTM × 2 → Linear → CTC). Only the CNN block differs. Gen1 is more accurate; gen2 is lighter. The building blocks for both live in `easyocr/model/modules.py`.

---

## 3. End-to-End Pipeline Overview

Here is the complete data flow from raw image list to decoded text:

```
easyocr.py
    └── get_text()                        recognition.py
            │
            ├── AlignCollate              # resize + pad → fixed-size batch tensor
            ├── ListDataset + DataLoader  # wrap image list for batched loading
            │
            └── recognizer_predict()
                    │
                    ├── model.forward()   # the neural network
                    │       │
                    │       ├── FeatureExtraction  (CNN)    [B,1,H,W]   → [B,512,1,W']
                    │       ├── Reshape bridge              [B,512,1,W']→ [B,W',512]
                    │       ├── SequenceModeling  (BiLSTM)  [B,W',512]  → [B,W',256]
                    │       └── Prediction        (Linear)  [B,W',256]  → [B,W',num_class]
                    │
                    ├── softmax + re-normalise (ignore separator chars)
                    ├── decode: greedy / beamsearch / wordbeamsearch
                    └── confidence score via custom_mean()
```

---

## 4. Stage 0 — Image Preprocessing

**File:** `easyocr/recognition.py` — classes `NormalizePAD` and `AlignCollate`

Before images can be fed to the neural network they must all be the same size, because PyTorch batches require uniform tensor shapes.

### Step 1: Resize to fixed height, preserve width

Each crop has a different aspect ratio. We resize every image to `H=32` pixels while scaling the width proportionally:

```
new_width = ceil(32 × original_width / original_height)
new_width  = min(new_width, imgW)   # cap at maximum (default 320 or 600)
```

The cap prevents very long crops from exceeding the model's design limit.

### Step 2: Pad to maximum width

After resizing, different crops still have different widths. `NormalizePAD` places each image on the left and fills the remaining columns on the right:

```
Pad_img[:, :, :w] = real_image
Pad_img[:, :, w:] = real_image[:, :, w-1]  ← last column repeated, not zero
```

Why repeat the last column instead of padding with black (zeros)?  
A hard black edge at the right side of a word would look like a vertical stroke to the CNN, potentially confusing it into predicting an extra character. Repeating the border pixel is a neutral, smooth continuation.

### Step 3: Normalise to [-1, 1]

```python
img = ToTensor(img)       # [1, H, W], float32 in [0, 1]
img.sub_(0.5).div_(0.5)   # rescale to [-1, 1]
```

This zero-centres the pixel values, which helps gradient flow during training and matches the distribution the model was trained on.

### Result

A batch tensor of shape `[B, 1, 32, imgW]` — all images the same height and width, grayscale, values in [-1, 1].

---

## 5. Stage 1 — Feature Extraction (CNN)

**File:** `easyocr/model/modules.py`

The CNN's job is to turn a 2D image into a **sequence of feature vectors** — one vector per horizontal "column slice" of the image. Each vector compactly encodes what the CNN saw in that vertical strip (strokes, curves, edges, textures).

The key challenge: the input has height 32 and variable width. The output must have height ≈ 1 (so it can be treated as a 1D sequence) while the width is preserved (so horizontal character positions survive).

### Generation 1: ResNet Backbone

**Class:** `ResNet_FeatureExtractor` → `ResNet` with `BasicBlock`, layers=`[1, 2, 5, 3]`

#### BasicBlock (Residual Block)

The building block of ResNet. Every block has a skip connection that adds the input directly to the output:

```
input ──► conv3×3 → BN → ReLU → conv3×3 → BN ──► (+) ──► ReLU ──► output
     └─────────────────────────────────────────────┘
                    skip (identity or 1×1 projection)
```

Why skip connections? They allow gradients to flow directly back through the network without vanishing, enabling much deeper networks to train stably. The `downsample` 1×1 conv is added only when input and output channel counts differ.

#### Full ResNet Spatial Schedule (input: `[B, 1, 32, 100]`)

| Stage | Operation | Output shape | Notes |
|-------|-----------|-------------|-------|
| Stem | conv3×3 (1→32), conv3×3 (32→64) | `[B, 64, 32, 100]` | Two small convs instead of one 7×7 |
| Stage 1 | MaxPool(2×2) + 1 BasicBlock (64→128) + conv3×3 | `[B, 128, 16, 50]` | H/2, W/2 |
| Stage 2 | MaxPool(2×2) + 2 BasicBlocks (128→256) + conv3×3 | `[B, 256, 8, 25]` | H/2, W/2 |
| Stage 3 | MaxPool((2,1),(2,1)) + 5 BasicBlocks (256→512) + conv3×3 | `[B, 512, 4, 26]` | **H/2, W unchanged** ← asymmetric |
| Stage 4 | 3 BasicBlocks (512→512) + conv(2,2,s=(2,1)) + conv(2,2,s=1) | `[B, 512, 1, 26]` | H collapses to 1 |

**Channel schedule:** 1 → 32 → 64 → 128 → 256 → 512 → 512 → 512

### Generation 2: VGG Backbone

**Class:** `VGG_FeatureExtractor`

Simpler architecture: plain Conv→ReLU→Pool blocks, no residual connections. Faster but slightly less accurate.

#### Full VGG Spatial Schedule (input: `[B, 1, 32, 100]`, output_channel=256)

| Block | Operations | Output shape | Notes |
|-------|-----------|-------------|-------|
| 1 | Conv(1→32) + ReLU + MaxPool(2×2) | `[B, 32, 16, 50]` | H/2, W/2 |
| 2 | Conv(32→64) + ReLU + MaxPool(2×2) | `[B, 64, 8, 25]` | H/2, W/2 |
| 3 | Conv(64→128) × 2 + ReLU + MaxPool((2,1),(2,1)) | `[B, 128, 4, 25]` | **H/2, W unchanged** |
| 4 | Conv(128→256)+BN × 2 + ReLU + MaxPool((2,1),(2,1)) | `[B, 256, 2, 25]` | **H/2, W unchanged** |
| 5 | Conv(256→256, kernel 2×1) + ReLU | `[B, 256, 1, 24]` | H: 2→1 |

**Channel schedule:** 1 → 32 → 64 → 128 → 128 → 256 → 256 → 256

### The Key Design Insight: Asymmetric Pooling

Standard image classification CNNs (e.g. ImageNet ResNet) reduce both height and width uniformly — that would destroy the horizontal position information that OCR needs.

EasyOCR's CNNs use `MaxPool(kernel=(2,1), stride=(2,1))` in the deeper stages:
- **Kernel height = 2, stride height = 2** → halves the height  
- **Kernel width = 1, stride width = 1** → leaves the width completely unchanged

This allows the CNN to "collapse" the 2D image into a 1D strip while keeping the width (and thus character positions) intact.

---

## 6. The Reshape Bridge: CNN → Sequence

**File:** `easyocr/model/model.py` (and `vgg_model.py`), in `forward()`

After the CNN outputs `[B, C, H'≈1, W']`, the tensor needs to become a sequence `[B, T, C]` where T = W' is the number of time steps.

```python
# Step 1: permute to bring W to position 1 (the "time" axis)
#   [B, C, H', W'] → [B, W', C, H']
visual_feature = visual_feature.permute(0, 3, 1, 2)

# Step 2: pool away any remaining height (in case H' > 1)
#   AdaptiveAvgPool2d((None, 1)): keep width dimension, pool height to 1
#   [B, W', C, H'] → [B, W', C, 1]
visual_feature = self.AdaptiveAvgPool(visual_feature)

# Step 3: remove the now-trivial height dimension
#   [B, W', C, 1] → [B, W', C]
visual_feature = visual_feature.squeeze(3)
```

After this bridge, `visual_feature` has shape `[B, W', 512]` (gen1) or `[B, W', 256]` (gen2).

Think of it as a sequence of `W'` vectors, where each vector is a 512-dimensional description of one vertical column of the original image.

---

## 7. Stage 2 — Sequence Modeling (BiLSTM)

**File:** `easyocr/model/modules.py` — class `BidirectionalLSTM`

**Input:** `[B, W', 512]`  
**Output:** `[B, W', 256]` (after two stacked BiLSTM layers)

### Why an LSTM at all?

The CNN features at each column are **local** — they describe what's visible in that narrow vertical strip. But OCR requires **global context**: the letter 'l' in isolation looks exactly like '1' or 'I', but in the context "hello" it is obviously 'l'.

An LSTM reads the sequence and builds a **hidden state** that accumulates information from all previous columns. This hidden state captures context like "we just saw 'hel', so this next shape is probably 'l' or 'o'".

### Why Bidirectional?

A standard left-to-right LSTM at column t knows only about columns 0..t. But in a word, future characters can disambiguate current ones too. A **bidirectional** LSTM runs two LSTMs:

```
Forward  LSTM: reads columns left → right  → h_fwd[t] knows about positions 0..t
Backward LSTM: reads columns right → left  → h_bwd[t] knows about positions t..T-1
```

Their hidden states are **concatenated**: `[h_fwd || h_bwd]` → shape `[B, T, 2×hidden]`.  
A `Linear(2×hidden → hidden)` then projects back to `hidden_size=256`.

### Why Two Stacked Layers?

```
Layer 1: CNN features [B, W', 512] → [B, W', 256]
         Translates raw visual features into contextual representations.

Layer 2: [B, W', 256] → [B, W', 256]
         Performs a second pass of temporal reasoning on the already-contextualised
         features, learning higher-order patterns (e.g. common letter combinations).
```

---

## 8. Stage 3 — Prediction Head (Linear)

**File:** `easyocr/model/model.py`

```python
self.Prediction = nn.Linear(hidden_size, num_class)
```

A single linear (fully-connected) layer applied **independently to each time step**:

```
Input:  [B, W', 256]
Output: [B, W', num_class]   ← raw logits, no softmax yet
```

`num_class = len(character) + 1` where `+1` is the **CTC blank token** at index 0.

For English, with ~96 characters: `num_class ≈ 97`.

The output can be thought of as: for each column of the image, a score distribution over "what character (or blank) does this column most likely correspond to?"

---

## 9. CTC Decoding

**File:** `easyocr/recognition.py` — `recognizer_predict()`  
**File:** `easyocr/utils.py` — class `CTCLabelConverter`

### What is CTC?

**Connectionist Temporal Classification** solves the alignment problem: the model outputs one score per column (say 26 scores for a 100px wide image), but the actual word might only be 5 characters long. CTC doesn't require you to say which column maps to which character — it learns the alignment automatically.

CTC introduces a special **blank token** (index 0) meaning "no character here, just filler." The decoding rules are:

1. Start with the raw sequence of argmax class indices per time step.
2. Collapse consecutive duplicate non-blank tokens into one.
3. Remove all blank tokens.

Example:
```
Raw sequence:   [h, h, blank, e, l, l, l, blank, l, o, blank]
After collapse: [h, blank, e, l, blank, l, o, blank]
After deblank:  [h, e, l, l, o]   → "hello"
```

### The CTCLabelConverter

`CTCLabelConverter` maps between characters and integer indices:

```python
character = "abcdefg..."          # string of supported characters
self.dict = {'a':1, 'b':2, ...}   # index 0 is RESERVED for blank
self.character = ['[blank]', 'a', 'b', 'c', ...]
```

The blank is always at index 0. All real characters are shifted to indices 1..N.

### Before Decoding: Softmax + Re-normalisation

```python
preds_prob = F.softmax(preds, dim=2)   # [B, T, C], values sum to 1 per time step
preds_prob[:, :, ignore_idx] = 0.      # zero out separator/boundary tokens
pred_norm = preds_prob.sum(axis=2)     # renormalise so remaining probs sum to 1
preds_prob = preds_prob / pred_norm
```

`ignore_idx` contains index 0 (blank) plus any language-separator tokens. These are zeroed out so they can never "win" the argmax during decoding, preventing them from appearing in the final text.

### Greedy Decoding

The simplest and fastest approach. At each time step, take `argmax`:

```python
_, preds_index = preds_prob.max(dim=2)   # [B, T] — index of most probable class
```

Then apply CTC collapse rules (remove consecutive duplicates, remove blanks).

```python
def decode_greedy(self, text_index, length):
    texts = []
    index = 0
    for l in length:
        t = text_index[index:index + l]
        # Boolean mask: keep positions where (a) current ≠ previous, OR (b) current ≠ blank
        # In NumPy terms: a = (t[1:] != t[:-1]), b = (t[1:] != 0)
        # This collapses consecutive duplicates and removes blanks in one step.
        char_list = [self.character[c] for i, c in enumerate(t)
                     if c != 0 and (i == 0 or c != t[i-1])]
        texts.append(''.join(char_list))
        index += l
    return texts
```

### Beam Search Decoding

More accurate but slower. Maintains `beamWidth` candidate sequences simultaneously. At each time step, extends each beam with the most probable next characters, keeping only the top `beamWidth` beams by total probability.

```
Time step 1: beams = [(h, 0.9), (H, 0.05), ...]
Time step 2: extend each → [(he, 0.81), (ha, 0.09), (He, ...), ...]
             → keep top beamWidth
...
Final: return highest-probability beam after CTC collapse
```

The beam search also supports an optional **language model** (bigram probability) to bias toward valid character sequences.

### Word Beam Search Decoding

An extension of beam search that uses a **word dictionary**. After finding the top candidate beams, it checks whether the decoded text appears in the loaded dictionary. If a dictionary word is found, it is preferred over a non-word candidate. Useful for improving accuracy on known vocabulary at the cost of creativity.

---

## 10. Confidence Scoring

**File:** `easyocr/recognition.py` — `custom_mean()` and `recognizer_predict()`

After decoding, a confidence score is computed for each prediction.

### Step 1: Collect per-character probabilities

For each time step, we already know which class "won" (the argmax index). We collect the probability of that winning class, **but only for non-blank time steps**:

```python
max_probs = v[i != 0]   # v = winning probabilities, i = winning indices
                         # i != 0 filters out blank tokens
```

This gives a list of probabilities — one per character position that the model "committed to" a real character.

### Step 2: Aggregate with custom_mean

```python
def custom_mean(x):
    return x.prod() ** (2.0 / np.sqrt(len(x)))
```

This is a modified geometric mean. For a 4-character prediction with all probs = 0.9:
```
Standard geometric mean: 0.9^4 ^ (1/4) = 0.9        (always returns ~0.9 regardless of length)
custom_mean:             0.9^4 ^ (2/2) = 0.9^4 ≈ 0.656  (longer words score lower)
```

The exponent `2/sqrt(n)` is a compromise: it's weaker than the full `1/n` geometric mean, so longer predictions are penalised (reflecting genuine added uncertainty) but not as harshly as a product would penalise them.

The resulting score is in `[0, 1]`. A score above ~0.5 is typically a reliable prediction.

---

## 11. Two-Pass Contrast Strategy

**File:** `easyocr/recognition.py` — `get_text()`

Some text crops are low-contrast: faint text on a similar-coloured background. The model may be uncertain on these, giving a low confidence score.

The solution is a **two-pass approach**:

```
Pass 1: Run recognition on all crops at normal contrast.
         → result1 = [(text, confidence), ...]

Identify low-confidence predictions (confidence < contrast_ths, default 0.1).

Pass 2: For low-confidence crops only, re-run with contrast enhancement.
         adjust_contrast_grey() linearly stretches pixel values so [low, high] → [0, 200].
         → result2 = [(text, confidence), ...]

Merge: for each low-confidence crop, keep whichever pass gave higher confidence.
```

The contrast enhancement function:

```python
def adjust_contrast_grey(img, target=0.4):
    contrast = (p90 - p10) / (p90 + p10)   # contrast measure
    if contrast < target:
        ratio = 200 / (high - low)
        img = (img - low + 25) * ratio      # shift and scale
        img = clip(img, 0, 255)             # clamp to valid range
    return img
```

Only crops that "failed" the first pass are re-processed, so the overhead is small.

---

## 12. Model Loading and Quantization

**File:** `easyocr/recognition.py` — `get_recognizer()`

### Loading weights from a DataParallel checkpoint

EasyOCR pretrained weights were saved from models wrapped in `torch.nn.DataParallel` (multi-GPU training). DataParallel prefixes every parameter name with `"module."`. When loading on a single CPU/GPU without DataParallel, the keys don't match.

```python
# CPU path: strip the 'module.' prefix manually
for key, value in state_dict.items():
    new_key = key[7:]            # "module.FeatureExtraction...." → "FeatureExtraction...."
    new_state_dict[new_key] = value

# GPU path: wrap in DataParallel first so the prefix matches
model = torch.nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(model_path))
```

### Dynamic Quantization (CPU only)

```python
torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
```

Replaces the `Linear` and `LSTM` weight matrices with int8 representations at inference time:
- **Model size:** ~4× smaller on disk
- **Speed:** 30–50% faster on CPU (integer arithmetic is cheaper)
- **Accuracy:** negligible loss for OCR tasks
- **GPU:** not used (GPU is already fast enough in float16/float32)

---

## 13. Complete Tensor Shape Walkthrough

Tracing a single batch of 8 images through the **gen1 (ResNet)** model with `imgH=32, imgW=100`:

```
Input crops (raw)
  → ListDataset + AlignCollate
  shape: [8, 1, 32, 100]    (B=8, C=1, H=32, W=100)

FeatureExtraction (ResNet)
  Stem:     [8, 1, 32, 100] → [8, 64, 32, 100]
  Stage 1:  MaxPool(2×2)    → [8, 64, 16, 50]
            1 BasicBlock    → [8, 128, 16, 50]
  Stage 2:  MaxPool(2×2)    → [8, 128, 8, 25]
            2 BasicBlocks   → [8, 256, 8, 25]
  Stage 3:  MaxPool((2,1))  → [8, 256, 4, 26]   ← asymmetric: H/2, W+1
            5 BasicBlocks   → [8, 512, 4, 26]
  Stage 4:  3 BasicBlocks   → [8, 512, 4, 26]
            conv4_1         → [8, 512, 2, 27]
            conv4_2         → [8, 512, 1, 26]

Reshape Bridge
  permute(0,3,1,2):         [8, 512, 1, 26] → [8, 26, 512, 1]
  AdaptiveAvgPool:          [8, 26, 512, 1] → [8, 26, 512, 1]  (H already 1)
  squeeze(3):               [8, 26, 512, 1] → [8, 26, 512]

BiLSTM Layer 1
  LSTM(512→256, bidir):     [8, 26, 512] → [8, 26, 512]  (bidir: 2×256)
  Linear(512→256):          [8, 26, 512] → [8, 26, 256]

BiLSTM Layer 2
  LSTM(256→256, bidir):     [8, 26, 256] → [8, 26, 512]
  Linear(512→256):          [8, 26, 512] → [8, 26, 256]

Linear Prediction Head
  Linear(256→num_class):    [8, 26, 256] → [8, 26, 97]

Softmax + Re-normalise
                            [8, 26, 97] → probabilities in [0,1] summing to 1

Greedy Decoding
  argmax per time step:     [8, 26]     → integer index per step per sample
  CTC collapse:             [8, 26]     → ["hello", "world", ...]

Confidence
  per-character probs:      list of float arrays (one per sample)
  custom_mean:              → [0.94, 0.87, ...]
```

---

## 14. Key Files Reference

| File | Role |
|------|------|
| `easyocr/recognition.py` | Entry point: `get_text()`, `get_recognizer()`, `recognizer_predict()`, `AlignCollate`, `NormalizePAD` |
| `easyocr/model/model.py` | Gen1 `Model` class (ResNet backbone) |
| `easyocr/model/vgg_model.py` | Gen2 `Model` class (VGG backbone) |
| `easyocr/model/modules.py` | `BidirectionalLSTM`, `ResNet_FeatureExtractor`, `ResNet`, `BasicBlock`, `VGG_FeatureExtractor` |
| `easyocr/utils.py` | `CTCLabelConverter`: character↔index mapping, greedy/beam/wordbeam decoding |
| `easyocr/config.py` | Per-language model config: which generation, character set, network parameters |
| `easyocr/easyocr.py` | Top-level `Reader` class that wires detection + recognition together |

---

## Summary

The EasyOCR recognition model is a **CRNN** (Convolutional Recurrent Neural Network):

1. **Preprocessing** — crops are resized to H=32, padded to fixed width, normalised to [-1,1].
2. **CNN** (ResNet or VGG) — extracts local visual features while collapsing height to 1 via asymmetric pooling, preserving horizontal character positions.
3. **Reshape** — converts the 2D feature map into a 1D sequence of column vectors.
4. **BiLSTM × 2** — adds left-and-right context across the sequence, enabling disambiguation of visually similar characters.
5. **Linear head** — scores each class at each time step (raw CTC logits).
6. **CTC decoding** — maps the per-column class scores to a final text string without requiring explicit character-to-column alignment.
7. **Confidence** — a modified geometric mean of the per-character peak probabilities.
8. **Two-pass contrast** — low-confidence crops are re-run with contrast enhancement for a second chance.
