import os
import sys
import re
import six
import math
import torch
import pandas as pd

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
try:
    from torch._utils import _accumulate          # older PyTorch (< 1.11)
except ImportError:
    from itertools import accumulate as _accumulate  # stdlib replacement
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAST HELPERS  (same as in easyocr/recognition.py)
# ─────────────────────────────────────────────────────────────────────────────

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/(high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    # Linearly stretches pixel values so low-contrast crops become more legible.
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./(high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

# ─────────────────────────────────────────────────────────────────────────────
# BATCH-BALANCED DATASET
# ─────────────────────────────────────────────────────────────────────────────

class Batch_Balanced_Dataset(object):
    # Ensures that every training batch contains a fixed proportion of samples
    # from each dataset source.
    #
    # Problem without this: if dataset A has 1M samples and dataset B has 10K,
    # training on their union means B contributes only 1% of batches and the
    # model never learns B's patterns well.
    #
    # Solution: create one separate DataLoader per source, and at each step pull
    # exactly (batch_ratio × batch_size) samples from each loader.
    #
    # Example: select_data="MJ-ST", batch_ratio="0.5-0.5", batch_size=192
    #   → 96 samples from MJ + 96 from ST every step, regardless of dataset sizes.

    def __init__(self, opt):
        log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0

        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            # Round to at least 1 so even tiny ratios always contribute something
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')

            # hierarchical_dataset walks the root folder recursively and collects
            # all sub-folders whose name contains `selected_d`
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            # opt.total_data_usage_ratio lets you train on a subset (e.g. 0.2 = 20%)
            # without removing data from disk.  Useful for quick experiments.
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            # Keep a persistent iterator per loader so get_batch() just calls next()
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size   # update so downstream code sees true batch size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        # Pull one mini-batch from each source loader and concatenate them.
        # When a loader runs out of data it is re-created (wraps around),
        # so training can run for any number of iterations.
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                # Loader exhausted → restart from the beginning
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        # Concatenate along the batch dimension: list of [Bi, C, H, W] → [B, C, H, W]
        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts

# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def hierarchical_dataset(root, opt, select_data='/'):
    # Recursively walks `root` and collects every leaf folder whose path contains
    # one of the names in `select_data`.  Each leaf folder is expected to contain:
    #   - labels.csv  (columns: filename, words)
    #   - the actual image files referenced by labels.csv
    #
    # Returns a ConcatDataset of all matching OCRDataset objects.
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:    # leaf directory (no sub-folders)
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = OCRDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset, dataset_log

# ─────────────────────────────────────────────────────────────────────────────
# OCR DATASET  (single folder)
# ─────────────────────────────────────────────────────────────────────────────

class OCRDataset(Dataset):
    # Reads a single dataset folder that has a `labels.csv` with two columns:
    #   filename  — image file name (relative to the folder)
    #   words     — the ground-truth text string
    #
    # Filtering (unless opt.data_filtering_off):
    #   1. Drop samples whose label is longer than opt.batch_max_length.
    #      Long labels never fit in the fixed-length CTC sequence.
    #   2. Drop samples whose label contains characters not in opt.character.
    #      The model has no class for unknown characters.

    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        print(root)
        # The separator regex '^([^,]+),' handles filenames that contain commas.
        self.df = pd.read_csv(os.path.join(root,'labels.csv'), sep='^([^,]+),', engine='python',
                              usecols=['filename', 'words'], keep_default_na=False)
        self.nSamples = len(self.df)

        if self.opt.data_filtering_off:
            self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
        else:
            self.filtered_index_list = []
            for index in range(self.nSamples):
                label = self.df.at[index,'words']
                try:
                    if len(label) > self.opt.batch_max_length:
                        continue    # too long to encode
                except:
                    print(label)
                # Regex to find any character NOT in the allowed set
                out_of_char = f'[^{self.opt.character}]'
                if re.search(out_of_char, label.lower()):
                    continue        # contains unsupported character
                self.filtered_index_list.append(index)
            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # Map the filtered index back to the original CSV row
        index = self.filtered_index_list[index]
        img_fname = self.df.at[index,'filename']
        img_fpath = os.path.join(self.root, img_fname)
        label = self.df.at[index,'words']

        if self.opt.rgb:
            img = Image.open(img_fpath).convert('RGB')
        else:
            img = Image.open(img_fpath).convert('L')   # grayscale

        if not self.opt.sensitive:
            label = label.lower()   # case-insensitive mode

        # Strip any remaining out-of-vocabulary characters from the label
        # (double protection — some may have slipped through the filter)
        out_of_char = f'[^{self.opt.character}]'
        label = re.sub(out_of_char, '', label)

        return (img, label)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

class ResizeNormalize(object):
    # Simple fixed-size resize + normalise.  Used when keep_ratio_with_pad=False:
    # all images are stretched to exactly (imgW × imgH) regardless of aspect ratio.
    # Faster but distorts the text shape for very wide or very tall crops.

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)    # normalise to [-1, 1]
        return img


class NormalizePAD(object):
    # Aspect-ratio-preserving resize + right-pad.
    # The image is placed on the left; the right side is filled with the last
    # column of pixels (not black zeros) to avoid a hard edge artefact.

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size                    # (C, H, maxW)
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)        # [C, H, W], float32 in [0,1]
        img.sub_(0.5).div_(0.5)         # normalise to [-1, 1]
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img  # [C, H, maxW]


class AlignCollate(object):
    # PyTorch collate function that batches images of different widths into a
    # single fixed-size tensor.  Called by DataLoader for every mini-batch.
    #
    # keep_ratio_with_pad=True  → use NormalizePAD (preserves aspect ratio)
    # keep_ratio_with_pad=False → use ResizeNormalize (squashes to fixed size)

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, contrast_adjust=0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)    # separate images from labels

        if self.keep_ratio_with_pad:
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size

                # Contrast augmentation: randomly boost low-contrast crops.
                # Helps the model learn to handle faint/washed-out text.
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(image, target=self.contrast_adjust)
                    image = Image.fromarray(image, 'L')

                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW           # cap at max width
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            # Stack: list of [C, H, W] → [N, C, H, W]
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels    # labels is a tuple of strings

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def tensor2im(image_tensor, imtype=np.uint8):
    # Converts a normalised tensor [C, H, W] back to a uint8 numpy image.
    # Reverses the sub_(0.5).div_(0.5) normalisation: x*2*255 roughly.
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))   # grayscale → 3-channel
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
