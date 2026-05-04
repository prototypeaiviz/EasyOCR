from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
import importlib
from .utils import CTCLabelConverter
import math

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def custom_mean(x):
    # Geometric-ish mean: product(probs) ^ (2 / sqrt(n))
    # For a 4-char word with all probs=0.9: 0.9^4 ^ (2/2) = 0.9^4 ≈ 0.656
    # The exponent 2/sqrt(n) shrinks as the word gets longer, which prevents
    # long words from being systematically penalised too harshly compared to
    # short ones (a regular geometric mean would raise 0.9^n to the 1/n power,
    # always giving 0.9 regardless of length; this formula gives a value that
    # still slightly decreases with length, reflecting real uncertainty).
    return x.prod()**(2.0/np.sqrt(len(x)))

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE CONTRAST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def contrast_grey(img):
    # Measure contrast using the spread between the 90th and 10th percentile
    # of pixel intensities (robust estimate, ignores extreme outliers).
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    # Ratio in [0,1]: 0 = flat grey, 1 = pure black-white.
    # np.maximum(10, ...) avoids division by near-zero sums.
    return (high-low)/np.maximum(10, high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    # If the measured contrast is already above `target`, leave the image alone.
    # Otherwise linearly stretch the pixel range so that [low-25, high+25] maps
    # to [0, 255].  This helps the model see faint text on low-contrast crops.
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)   # scale factor
        img = (img - low + 25)*ratio             # shift then scale
        # clamp to [0, 255]
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — NormalizePAD
# ─────────────────────────────────────────────────────────────────────────────

class NormalizePAD(object):
    # Converts a PIL grayscale image to a fixed-size float tensor.
    # Steps:
    #   1. ToTensor  → shape [1, H, W], values in [0, 1]
    #   2. Normalise → subtract 0.5, divide by 0.5 → values in [-1, 1]
    #   3. Right-pad to max_size width by repeating the last column of pixels.
    #      (Repeating the border rather than zero-padding avoids a hard black
    #       edge that could confuse the CNN.)

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size                          # (C, H, maxW)
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)        # [1, H, W], float32 in [0,1]
        img.sub_(0.5).div_(0.5)         # normalise to [-1, 1]
        c, h, w = img.size()

        # Allocate a zero tensor of the target size
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img         # place the real image on the left
        if self.max_size[2] != w:
            # Fill the right padding with the last column of the real image
            # unsqueeze(2) → [C, H, 1], then expand to [C, H, pad_width]
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img  # shape: (C, H, maxW)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET — ListDataset
# ─────────────────────────────────────────────────────────────────────────────

class ListDataset(torch.utils.data.Dataset):
    # Minimal Dataset wrapper around a plain Python list of numpy arrays.
    # Each array is a single grayscale text crop (height=32, variable width).
    # __getitem__ converts the numpy array to a PIL Image in 'L' (greyscale) mode
    # so that AlignCollate can call .size, .resize, etc.

    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, 'L')    # numpy uint8 → PIL grayscale

# ─────────────────────────────────────────────────────────────────────────────
# COLLATE FUNCTION — AlignCollate
# ─────────────────────────────────────────────────────────────────────────────

class AlignCollate(object):
    # Custom collate function used by the DataLoader.
    # PyTorch's default collate expects all tensors in a batch to have the
    # same shape.  Text images have different widths, so we cannot stack them
    # directly.  AlignCollate:
    #   1. Resizes each image to height=imgH while keeping the aspect ratio.
    #   2. Caps the width at imgW.
    #   3. Pads all images to (imgH × imgW) via NormalizePAD.
    #   4. Stacks them into a single batch tensor [N, 1, imgH, imgW].

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, adjust_contrast = 0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast  # 0 = no adjustment

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)  # drop any None entries
        images = batch

        resized_max_w = self.imgW
        input_channel = 1   # grayscale
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size   # PIL: (width, height)

            # Optional second-pass contrast enhancement (used for low-confidence re-runs)
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L"))
                image = adjust_contrast_grey(image, target = self.adjust_contrast)
                image = Image.fromarray(image, 'L')

            # Preserve aspect ratio: new_w = imgH * (w/h), capped at imgW
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            # BICUBIC gives smoother edges than nearest-neighbour when downscaling
            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))  # → padded tensor [1, imgH, imgW]

        # Stack along batch dimension: list of [1, H, W] → [N, 1, H, W]
        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOOP — recognizer_predict
# ─────────────────────────────────────────────────────────────────────────────

def recognizer_predict(model, converter, test_loader, batch_max_length,\
                       ignore_idx, char_group_idx, decoder = 'greedy', beamWidth= 5, device = 'cpu'):
    # Runs the recognition model over all batches in test_loader and returns
    # a list of (predicted_string, confidence_score) tuples.

    model.eval()    # disable dropout / batch-norm training behaviour
    result = []
    with torch.no_grad():   # no gradient computation needed at inference
        for image_tensors in test_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)    # move to GPU if available

            # These tensors are required by the model signature but are not
            # used by the CTC-based model at inference time (they are only
            # relevant for attention-based models during teacher-forced decoding).
            length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)

            # Forward pass → raw logits: [batch, T, num_class]
            # T is the number of "time steps" (roughly image_width / 4 after CNN pooling)
            preds = model(image, text_for_pred)

            # CTC sequences are aligned to time steps, so preds_size tells the
            # CTC decoder how many time steps each sample has (all the same here).
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

            # ── Probability re-normalisation ─────────────────────────────────
            # 1. Softmax over class dimension → probabilities sum to 1 per time step
            preds_prob = F.softmax(preds, dim=2)            # [B, T, C]
            preds_prob = preds_prob.cpu().detach().numpy()

            # 2. Zero out ignored characters (separators between languages, etc.)
            #    so they cannot win argmax.
            preds_prob[:,:,ignore_idx] = 0.

            # 3. Re-normalise so the remaining probabilities still sum to 1
            pred_norm = preds_prob.sum(axis=2)              # [B, T]
            preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)

            preds_prob = torch.from_numpy(preds_prob).float().to(device)

            # ── Decoding ─────────────────────────────────────────────────────
            if decoder == 'greedy':
                # At each time step pick the most probable class index,
                # then collapse consecutive duplicates and remove blanks.
                _, preds_index = preds_prob.max(2)          # [B, T] argmax indices
                preds_index = preds_index.view(-1)          # flatten for batch-decode
                preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)
            elif decoder == 'beamsearch':
                k = preds_prob.cpu().detach().numpy()       # [B, T, C]
                preds_str = converter.decode_beamsearch(k, beamWidth=beamWidth)
            elif decoder == 'wordbeamsearch':
                k = preds_prob.cpu().detach().numpy()
                preds_str = converter.decode_wordbeamsearch(k, beamWidth=beamWidth)

            # ── Confidence score ─────────────────────────────────────────────
            preds_prob = preds_prob.cpu().detach().numpy()
            values = preds_prob.max(axis=2)         # best probability at each time step
            indices = preds_prob.argmax(axis=2)     # which class won at each time step

            preds_max_prob = []
            for v, i in zip(values, indices):
                # Keep only time steps where argmax was NOT the blank token (index 0).
                # These correspond to the actual predicted characters.
                max_probs = v[i!=0]
                if len(max_probs)>0:
                    preds_max_prob.append(max_probs)
                else:
                    preds_max_prob.append(np.array([0]))    # fallback for empty predictions

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                confidence_score = custom_mean(pred_max_prob)
                result.append([pred, confidence_score])

    return result   # list of [string, float] per input image

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER — get_recognizer
# ─────────────────────────────────────────────────────────────────────────────

def get_recognizer(recog_network, network_params, character,\
                   separator_list, dict_list, model_path,\
                   device = 'cpu', quantize = True):
    # Builds the recognition model and loads pretrained weights.
    #
    # recog_network:  'generation1' → ResNet backbone (model/model.py)
    #                 'generation2' → VGG backbone   (model/vgg_model.py)
    #                 anything else → treated as a dotted module path to import

    # CTCLabelConverter maps characters to integer indices and back.
    # num_class = number of unique characters + 1 (the CTC blank token).
    converter = CTCLabelConverter(character, separator_list, dict_list)
    num_class = len(converter.character)

    if recog_network == 'generation1':
        model_pkg = importlib.import_module("easyocr.model.model")
    elif recog_network == 'generation2':
        model_pkg = importlib.import_module("easyocr.model.vgg_model")
    else:
        model_pkg = importlib.import_module(recog_network)

    # Instantiate the model with its hyperparameters (input_channel, output_channel,
    # hidden_size, num_class) unpacked from network_params dict.
    model = model_pkg.Model(num_class=num_class, **network_params)

    if device == 'cpu':
        # The weights were saved from a DataParallel-wrapped model, so every key
        # has a 'module.' prefix (e.g. 'module.FeatureExtraction.ConvNet.conv0_1.weight').
        # Strip that prefix to match the plain model's parameter names.
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:]   # strip 'module.'
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

        if quantize:
            # Dynamic quantisation converts Linear and LSTM weights to int8 at
            # runtime, reducing model size ~4× and speeding up CPU inference.
            try:
                torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        # On GPU wrap in DataParallel so the 'module.' prefix in the checkpoint
        # matches the model's parameter names directly.
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    return model, converter

# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL ENTRY POINT — get_text
# ─────────────────────────────────────────────────────────────────────────────

def get_text(character, imgH, imgW, recognizer, converter, image_list,\
             ignore_char = '',decoder = 'greedy', beamWidth =5, batch_size=1, contrast_ths=0.1,\
             adjust_contrast=0.5, filter_ths = 0.003, workers = 1, device = 'cpu'):
    # Orchestrates the full recognition pipeline:
    #   1. Build DataLoaders from the cropped text images.
    #   2. First inference pass at normal contrast.
    #   3. For low-confidence results, run a second pass with contrast enhancement.
    #   4. Keep whichever pass gave the higher confidence.
    #
    # image_list: list of (bounding_box, numpy_crop) pairs produced by the detector.
    # imgW: maximum width in pixels that the recognition model accepts.
    # batch_max_length: hard cap on sequence length; imgW/10 ≈ one char per 10px column.

    batch_max_length = int(imgW/10)

    char_group_idx = {}
    ignore_idx = []
    for char in ignore_char:
        try: ignore_idx.append(character.index(char)+1)
        except: pass

    coord = [item[0] for item in image_list]    # bounding box coordinates
    img_list = [item[1] for item in image_list] # numpy grayscale crops

    # ── Pass 1: normal contrast ───────────────────────────────────────────────
    AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=int(workers), collate_fn=AlignCollate_normal, pin_memory=True)
    # pin_memory=True speeds up CPU→GPU transfer by keeping tensors in pinned RAM.

    result1 = recognizer_predict(recognizer, converter, test_loader, batch_max_length,\
                                 ignore_idx, char_group_idx, decoder, beamWidth, device = device)

    # ── Pass 2: contrast-enhanced (only for low-confidence predictions) ───────
    low_confident_idx = [i for i,item in enumerate(result1) if (item[1] < contrast_ths)]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        AlignCollate_contrast = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True, adjust_contrast=adjust_contrast)
        test_data = ListDataset(img_list2)
        test_loader = torch.utils.data.DataLoader(
                        test_data, batch_size=batch_size, shuffle=False,
                        num_workers=int(workers), collate_fn=AlignCollate_contrast, pin_memory=True)
        result2 = recognizer_predict(recognizer, converter, test_loader, batch_max_length,\
                                     ignore_idx, char_group_idx, decoder, beamWidth, device = device)

    # ── Merge: pick the better result for each image ──────────────────────────
    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1]>pred2[1]:
                result.append( (box, pred1[0], pred1[1]) )
            else:
                result.append( (box, pred2[0], pred2[1]) )
        else:
            result.append( (box, pred1[0], pred1[1]) )

    return result   # list of (bounding_box, text_string, confidence_float)
