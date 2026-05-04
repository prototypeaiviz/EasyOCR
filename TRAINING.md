# EasyOCR Recognition Model — Training & Fine-Tuning Guide

This document covers everything you need to train or fine-tune the EasyOCR **recognition** model from scratch or on your own data.  Detection model training (CRAFT / DBNet) is out of scope.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Repository layout](#2-repository-layout)
3. [Dataset format](#3-dataset-format)
4. [Config file](#4-config-file)
5. [All config parameters explained](#5-all-config-parameters-explained)
6. [How to run training](#6-how-to-run-training)
7. [Fine-tuning strategies](#7-fine-tuning-strategies)
8. [Understanding the training loop](#8-understanding-the-training-loop)
9. [Reading training output](#9-reading-training-output)
10. [Using your trained model with EasyOCR](#10-using-your-trained-model-with-easyocr)
11. [Common issues](#11-common-issues)

---

## 1. Architecture Overview

The recognition model is a **CRNN** (Convolutional Recurrent Neural Network).  Every text crop passes through four stages:

```
Cropped text image  [B, 1, H=32, W]
         │
  ┌──────▼──────────────────────────────────────┐
  │  Stage 0 — Transformation  (optional TPS)   │
  │  Rectifies curved / tilted text             │
  └──────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────┐
  │  Stage 1 — Feature Extraction  (CNN)         │
  │  VGG / RCNN / ResNet                         │
  │  Collapses image height → 1 row              │
  │  Output: [B, W', output_channel]             │
  └──────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────┐
  │  Stage 2 — Sequence Modeling  (BiLSTM)       │
  │  Reads the W' column-features left+right     │
  │  Output: [B, W', hidden_size]                │
  └──────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────┐
  │  Stage 3 — Prediction                        │
  │  CTC  : Linear → raw logits [B, W', vocab]   │
  │  Attn : attention decoder, one char/step     │
  └──────────────────────────────────────────────┘
         │
   CTC / Attention decoder
         │
   "Hello"  + confidence
```

**Two model generations exist in EasyOCR inference:**

| Generation | CNN backbone | Config flag |
|---|---|---|
| gen1 | Custom ResNet `[1,2,5,3]` BasicBlocks | `FeatureExtraction: 'ResNet'` |
| gen2 | VGG-style (simpler, faster) | `FeatureExtraction: 'VGG'` |

Both use 2× stacked BiLSTM + CTC by default.

---

## 2. Repository Layout

```
EasyOCR/
├── trainer/
│   ├── trainer.ipynb          ← ENTRY POINT: open this to start training
│   ├── train.py               ← main training function  train(opt)
│   ├── test.py                ← validation function     validation(...)
│   ├── model.py               ← flexible 4-stage Model class
│   ├── dataset.py             ← OCRDataset, AlignCollate, Batch_Balanced_Dataset
│   ├── utils.py               ← CTCLabelConverter, AttnLabelConverter, Averager
│   ├── config_files/
│   │   └── en_filtered_config.yaml   ← example config (copy & edit this)
│   └── modules/
│       ├── transformation.py  ← TPS spatial transformer
│       ├── feature_extraction.py  ← VGG / RCNN / ResNet CNNs
│       ├── sequence_modeling.py   ← BidirectionalLSTM
│       └── prediction.py          ← Attention decoder
│
├── easyocr/
│   ├── model/
│   │   ├── model.py           ← gen1 inference model (ResNet)
│   │   └── vgg_model.py       ← gen2 inference model (VGG)
│   └── recognition.py         ← inference pipeline
│
└── saved_models/              ← created automatically, checkpoints go here
    └── <experiment_name>/
        ├── best_accuracy.pth
        ├── best_norm_ED.pth
        ├── iter_10000.pth
        ├── log_train.txt
        ├── log_dataset.txt
        └── opt.txt
```

---

## 3. Dataset Format

### Folder structure

```
all_data/
├── my_dataset/
│   ├── labels.csv
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── my_val/
    ├── labels.csv
    └── ...
```

- `train_data` in the config points to `all_data/`
- `select_data` names the sub-folder(s) to use, e.g. `my_dataset`
- `valid_data` points to the validation folder, e.g. `all_data/my_val`

### labels.csv format

Two columns, comma-separated:

```
filename,words
img_001.jpg,Hello
img_002.jpg,World
img_003.jpg,EasyOCR
```

- `filename` — image file name relative to the same folder
- `words`    — the ground-truth text string for that crop

**Important rules:**
- Every image should already be cropped to a single line of text.
- Images should be grayscale or RGB (set `rgb: True` for RGB).
- Recommended height: at least 32 pixels (the model will resize to `imgH`).
- Labels longer than `batch_max_length` characters are silently dropped during loading.
- Characters not in `opt.character` are stripped from labels at load time.

### Mixing multiple datasets

Set `select_data` and `batch_ratio` as dash-separated lists of equal length:

```yaml
select_data:  'dataset_a-dataset_b'
batch_ratio:  '0.6-0.4'
```

This puts 60 % of each batch from `dataset_a` and 40 % from `dataset_b`, regardless of their relative sizes.  The dataset with the larger ratio will be sampled with repetition once exhausted.

---

## 4. Config File

Copy `trainer/config_files/en_filtered_config.yaml` and edit it for your task.  The notebook reads this file and passes it as `opt` to `train()`.

**Minimal config for fine-tuning on English:**

```yaml
# ── Character set ──────────────────────────────────────────────────────────
number:    '0123456789'
symbol:    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €"
lang_char: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# ── Experiment ─────────────────────────────────────────────────────────────
experiment_name: 'my_finetune'

# ── Data ───────────────────────────────────────────────────────────────────
train_data:  'all_data'
valid_data:  'all_data/my_val'
select_data: 'my_dataset'
batch_ratio: '1'

# ── Training ───────────────────────────────────────────────────────────────
saved_model: 'saved_models/english_g2/best_accuracy.pth'   # pretrained weights
FT:          True        # fine-tune (strict=False)
num_iter:    50000
valInterval: 2000
batch_size:  32

# ── Image preprocessing ────────────────────────────────────────────────────
imgH:             64
imgW:             600
PAD:              True    # keep aspect ratio + pad
batch_max_length: 34

# ── Model architecture ─────────────────────────────────────────────────────
Transformation:   'None'
FeatureExtraction: 'VGG'
SequenceModeling: 'BiLSTM'
Prediction:       'CTC'
input_channel:    1
output_channel:   256
hidden_size:      256

# ── Optimizer ──────────────────────────────────────────────────────────────
optim:     False   # False = Adadelta (recommended)
lr:        1.0
rho:       0.95
eps:       1.0e-8
grad_clip: 5
```

---

## 5. All Config Parameters Explained

### Character set

| Key | What it does |
|---|---|
| `number` | Digit characters included in the vocabulary |
| `symbol` | Punctuation and special characters |
| `lang_char` | Language-specific alphabet characters |
| `lang_char: 'None'` | Auto-detect characters from the CSV labels |

The final `opt.character` = `number + symbol + lang_char`.  The vocabulary size = `len(character) + 1` (the +1 is the CTC blank token).

### Experiment & paths

| Key | What it does |
|---|---|
| `experiment_name` | Sub-folder created under `saved_models/` for logs and checkpoints |
| `train_data` | Root folder containing all training dataset sub-folders |
| `valid_data` | Path to validation dataset folder |
| `saved_model` | Path to a `.pth` checkpoint to load before training |
| `FT` | `True` = fine-tune mode (`strict=False`, tolerates missing/extra keys) |
| `new_prediction` | `True` = replace the Prediction head with a freshly initialised one (needed when `saved_model` was trained on a different vocab size) |

### Data loading

| Key | What it does |
|---|---|
| `select_data` | Dash-separated list of dataset sub-folder names to use |
| `batch_ratio` | Matching dash-separated fractions (must sum to ~1) |
| `total_data_usage_ratio` | Fraction of each dataset to actually load (e.g. `0.5` uses 50 %) |
| `batch_max_length` | Maximum number of characters per label; longer samples are filtered out |
| `data_filtering_off` | `True` = skip label filtering (use all samples as-is) |
| `workers` | Number of DataLoader worker processes |
| `manualSeed` | Random seed for reproducibility |

### Image preprocessing

| Key | What it does |
|---|---|
| `imgH` | Target image height in pixels (model input height). Typically 32 or 64 |
| `imgW` | Maximum image width in pixels. Wider crops are squashed to this width |
| `PAD` | `True` = preserve aspect ratio and right-pad. `False` = stretch to (imgH × imgW) |
| `rgb` | `True` = 3-channel RGB input. `False` = grayscale (recommended) |
| `contrast_adjust` | Contrast enhancement target during data loading (0 = disabled, 0.5 = typical). Applied at training time as augmentation |
| `sensitive` | `True` = case-sensitive labels. `False` = lowercase all labels |

### Training loop

| Key | What it does |
|---|---|
| `num_iter` | Total number of training iterations (not epochs) |
| `valInterval` | Run validation every N iterations |
| `batch_size` | Total batch size (sum across all `batch_ratio` sources) |

### Model architecture

| Key | Options | Notes |
|---|---|---|
| `Transformation` | `'None'`, `'TPS'` | TPS adds a spatial rectification stage before the CNN. Improves curved/rotated text but slower |
| `FeatureExtraction` | `'VGG'`, `'RCNN'`, `'ResNet'` | VGG = fastest, ResNet = most accurate, RCNN = in between |
| `SequenceModeling` | `'BiLSTM'`, `'None'` | BiLSTM adds left-right context. Almost always use BiLSTM |
| `Prediction` | `'CTC'`, `'Attn'` | CTC = faster & simpler. Attn = sometimes better for short text |
| `num_fiducial` | integer (default 20) | Number of TPS control points (only used when `Transformation: 'TPS'`) |
| `input_channel` | 1 or 3 | Must match `rgb` setting |
| `output_channel` | integer (default 256) | CNN output depth. 256 for gen2/VGG, 512 for gen1/ResNet |
| `hidden_size` | integer (default 256) | LSTM hidden dimension |
| `decode` | `'greedy'`, `'beamsearch'` | Decoder used during validation |

### Optimizer

| Key | What it does |
|---|---|
| `optim` | `False` = Adadelta (default, no LR tuning needed). `True` = Adam |
| `lr` | Learning rate. For Adadelta: 1.0. For Adam: try 1e-4 |
| `rho` | Adadelta decay factor (default 0.95) |
| `eps` | Adadelta epsilon (default 1e-8) |
| `beta1` | Adam beta1 (default 0.9, only used when `optim: True`) |
| `grad_clip` | Maximum gradient norm (prevents LSTM exploding gradients) |

### Fine-tuning / layer freezing

| Key | What it does |
|---|---|
| `freeze_FeatureFxtraction` | `True` = freeze CNN weights (only LSTM + head are trained) |
| `freeze_SequenceModeling` | `True` = freeze BiLSTM weights (only prediction head is trained) |

---

## 6. How to Run Training

### Step 1 — Install dependencies

```bash
pip install easyocr torch torchvision natsort pandas nltk pyyaml
```

### Step 2 — Prepare your dataset

Create a folder structure as described in [Section 3](#3-dataset-format) and write a `labels.csv` in each leaf folder.

### Step 3 — Create a config file

Copy `trainer/config_files/en_filtered_config.yaml` to e.g. `trainer/config_files/my_config.yaml` and edit it.

### Step 4 — Run via the notebook

Open `trainer/trainer.ipynb` in Jupyter and run all cells.  The notebook:
1. Loads the YAML config into an `AttrDict` (`opt`).
2. Builds the character set from `number + symbol + lang_char`.
3. Creates the `saved_models/<experiment_name>/` folder.
4. Calls `train(opt, amp=False)`.

To enable **mixed-precision training** (faster on modern GPUs):
```python
train(opt, amp=True)
```

### Step 5 — Monitor progress

```
saved_models/my_finetune/log_train.txt
```

Each `valInterval` steps you will see a line like:
```
[2000/50000] Train loss: 0.12345, Valid loss: 0.23456, Elapsed_time: 183.4
Current_accuracy   : 87.650, Current_norm_ED   : 0.9421
Best_accuracy      : 87.650, Best_norm_ED      : 0.9421
```

---

## 7. Fine-Tuning Strategies

### Strategy A — Full fine-tune (same vocab)

Use when you have a pretrained model for the same language but want to adapt it to a new domain (e.g. documents, handwriting, street signs).

```yaml
saved_model: 'path/to/pretrained.pth'
FT:          False   # strict=True: every key must match exactly
num_iter:    30000
```

### Strategy B — Fine-tune with partial weight reuse (`FT: True`)

Use when some architecture details differ slightly or when you want to skip strict key matching.

```yaml
saved_model: 'path/to/pretrained.pth'
FT:          True    # strict=False: silently ignores mismatched keys
```

### Strategy C — New character set / new language

Use when you are adding characters the pretrained model never saw, which requires a new Prediction head with a different output size.

```yaml
saved_model:    'path/to/pretrained.pth'
FT:             True
new_prediction: True   # replaces the Prediction linear layer with a fresh one
```

Typical progression for a new language:
1. Start with `new_prediction: True`, train for ~50k iterations.
2. Once loss stabilises, set `new_prediction: False` and continue training the whole model.

### Strategy D — Freeze CNN, train LSTM + head only

Use when your new data is stylistically similar to the pretrained data (the CNN features transfer well) but the sequence patterns differ.  Freezing the CNN speeds up training ~2× and prevents overfitting on small datasets.

```yaml
saved_model:              'path/to/pretrained.pth'
FT:                       True
freeze_FeatureFxtraction: True
freeze_SequenceModeling:  False
```

### Strategy E — Freeze everything except the head

Use when you have very few labelled samples (<5 000) and only need to teach the model new characters.

```yaml
freeze_FeatureFxtraction: True
freeze_SequenceModeling:  True
# new_prediction: True   # only if vocab size changes
```

### Strategy F — Train from scratch

Leave `saved_model` empty and set all freeze flags to `False`.  Use a large dataset (>100k samples) and more iterations (300k+).

```yaml
saved_model: ''
num_iter:    300000
```

### Choosing `num_iter`

| Scenario | Suggested iterations |
|---|---|
| Full fine-tune, large dataset (>100k) | 100k – 300k |
| Fine-tune, medium dataset (10k–100k) | 30k – 100k |
| Fine-tune, small dataset (<10k) | 10k – 30k |
| New prediction head only | 10k – 50k |
| Training from scratch | 300k+ |

Checkpoints are saved every 10 000 iterations plus whenever `best_accuracy.pth` or `best_norm_ED.pth` is updated.

---

## 8. Understanding the Training Loop

```
for each iteration:
    1.  optimizer.zero_grad()

    2.  image_tensors, labels = train_dataset.get_batch()
        # get_batch() pulls the right number of samples from each
        # dataset source according to batch_ratio

    3.  image = image_tensors.to(device)
        text, length = converter.encode(labels)
        # encode: 'cat' → [3,1,20] (integer indices), length=[3]

    4.  if CTC:
            preds = model(image, text)           # [B, T, vocab]
            preds = preds.log_softmax(2)
            preds = preds.permute(1, 0, 2)       # [T, B, vocab]  ← CTCLoss format
            cost  = CTCLoss(preds, text, preds_size, length)

        if Attn:
            preds  = model(image, text[:, :-1])  # teacher-forced
            target = text[:, 1:]                 # [GO] stripped
            cost   = CrossEntropyLoss(preds, target)

    5.  cost.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    6.  every valInterval steps:
            run validation → log accuracy + norm-ED
            save best_accuracy.pth and best_norm_ED.pth
```

### What CTCLoss actually measures

CTC does not require character-level alignment between the image and the label.  It sums the probabilities of **all valid paths** through the output sequence that decode to the target string, and maximises that sum.  The model learns which CNN column corresponds to which character implicitly.

### What Attention loss measures

At each decoding step `i`, the model sees the ground-truth token `text[:,i]` (teacher-forcing) and predicts `text[:,i+1]`.  CrossEntropyLoss is computed per token.

---

## 9. Reading Training Output

### Loss values

- **Train loss** — averaged over the last `valInterval` iterations.  Should decrease steadily.
- **Valid loss** — computed on the validation set.  If this goes up while train loss goes down, the model is overfitting.

### Accuracy metrics

- **Current_accuracy** — percentage of validation samples where the predicted string exactly matches the ground truth (including punctuation and case).
- **Current_norm_ED** — ICDAR2019 Normalised Edit Distance, averaged over the dataset.  Score of 1.0 = all predictions are perfect.  Score of 0.9 = predictions have on average 10 % of their characters wrong.

`best_accuracy.pth` and `best_norm_ED.pth` are saved independently — the best-ED model is often the more useful one because it rewards partially-correct predictions.

### Sample predictions table

```
Ground Truth              | Prediction               | Confidence Score & T/F
--------------------------------------------------------------------------------
Hello World               | Hello World              | 0.9876   True
2024-05-01                | 2024-05-01               | 0.8932   True
Illegible crop            | lllegnle crop            | 0.1234   False
```

Confidence is the geometric mean of per-character max probabilities.  Low confidence (<0.3) usually means a blurry crop or a character the model hasn't learned yet.

---

## 10. Using Your Trained Model with EasyOCR

### Option A — Pass the `.pth` path directly

```python
import easyocr

reader = easyocr.Reader(
    ['en'],
    recog_network='generation2',       # or 'generation1' if you used ResNet
    model_storage_directory='./models',
    user_network_directory='./models',
)

# Monkey-patch the recognizer with your weights:
from easyocr.recognition import get_recognizer
from easyocr.config import recognition_models

network_params = {
    'input_channel': 1,
    'output_channel': 256,
    'hidden_size': 256,
}
character = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

recognizer, converter = get_recognizer(
    recog_network='generation2',
    network_params=network_params,
    character=character,
    separator_list={},
    dict_list={},
    model_path='saved_models/my_finetune/best_accuracy.pth',
    device='cpu',
    quantize=True,
)

reader.recognizer = recognizer
reader.converter  = converter

result = reader.readtext('test_image.jpg')
```

### Option B — Register as a custom model

Place your `.pth` file in `~/.EasyOCR/model/` and follow the EasyOCR custom model registration guide to add an entry to `config.py`.

### Matching trainer and inference architectures

| Trainer config | Inference model |
|---|---|
| `FeatureExtraction: 'VGG'` | `recog_network='generation2'`  (`easyocr/model/vgg_model.py`) |
| `FeatureExtraction: 'ResNet'` | `recog_network='generation1'` (`easyocr/model/model.py`) |

The `network_params` dict must match what you trained with:

```python
# For VGG gen2 with output_channel=256, hidden_size=256
network_params = {'input_channel': 1, 'output_channel': 256, 'hidden_size': 256}

# For ResNet gen1 with output_channel=512, hidden_size=256
network_params = {'input_channel': 1, 'output_channel': 512, 'hidden_size': 256}
```

---

## 11. Common Issues

### Loss does not decrease

- **Learning rate too high** — try reducing `lr` from 1.0 to 0.1 for Adadelta, or use Adam with `lr: 0.0001`.
- **Batch too small** — for CTC, batches of 16+ are usually needed for stable gradient estimates.
- **Labels have out-of-vocab characters** — check that `character` covers all characters in your CSV.  Enable `data_filtering_off: False` to filter them automatically.

### Accuracy stuck at 0 for many iterations

- Normal for the first few thousand iterations when training from scratch or with a new prediction head (`new_prediction: True`).
- Check that labels are correctly formatted: no BOM characters, consistent encoding (UTF-8).
- If using `Prediction: 'CTC'`, make sure `batch_max_length` ≥ your longest label.

### CUDA out of memory

- Reduce `batch_size`.
- Reduce `output_channel` (e.g. 256 → 128) or `hidden_size` (e.g. 256 → 128).
- Enable AMP: `train(opt, amp=True)`.
- Set `workers: 0` to disable multi-process data loading.

### `ImportError: cannot import name '_accumulate'`

Fixed in `dataset.py` — this was a PyTorch version incompatibility.

### Checkpoint shape mismatch on load

- If `FT: False` and the vocab size or architecture changed, PyTorch will error with a size mismatch.
- Use `FT: True` (strict=False) to load weights for the matching layers only.
- Use `new_prediction: True` if only the final linear layer has a different size.

### Low confidence on short words

The confidence score uses `prod(probs)^(2/sqrt(n))`.  Words with 1–2 characters naturally get lower scores because there are fewer per-character probabilities to multiply.  This is expected behaviour, not a sign of poor accuracy.

### Training/validation split

Use a held-out set that is representative of your real inference images.  If your validation set is too easy (e.g. clean printed text) and your real data is noisy, `best_accuracy.pth` will not be the best model for production.  Consider building a validation set that matches your deployment conditions.
