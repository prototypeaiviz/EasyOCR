import os
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model

def validation(model, criterion, evaluation_loader, converter, opt, device):
    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION / EVALUATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    # Called from train.py every opt.valInterval steps.
    # Returns two accuracy metrics:
    #
    #   accuracy    — percentage of samples where the predicted string exactly
    #                 matches the ground truth (case-sensitive unless opt.sensitive=False)
    #
    #   norm_ED     — ICDAR2019 Normalised Edit Distance, averaged over the dataset.
    #                 For each sample:
    #                   if len(gt) > len(pred): 1 - edit_distance / len(gt)
    #                   else:                   1 - edit_distance / len(pred)
    #                 Score in [0,1]; 1 = perfect match, 0 = completely wrong.
    #                 More informative than exact accuracy for long or rare words.

    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        # Dummy tensors — needed by the model API but not used by CTC at inference
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        # Ground-truth text encoded as integer indices (needed to compute the loss)
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()

        if 'CTC' in opt.Prediction:
            # ── CTC branch ───────────────────────────────────────────────────
            preds = model(image, text_for_pred)         # [B, T, num_class]
            forward_time = time.time() - start_time

            # CTCLoss expects log-probabilities in [T, B, C] format.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            cost = criterion(preds.log_softmax(2).permute(1, 0, 2),
                             text_for_loss, preds_size, length_for_loss)

            if opt.decode == 'greedy':
                # Argmax per time step, then collapse duplicates + remove blanks
                _, preds_index = preds.max(2)       # [B, T]
                preds_index = preds_index.view(-1)  # flatten for batch-decode
                preds_str = converter.decode_greedy(preds_index.data, preds_size.data)
            elif opt.decode == 'beamsearch':
                preds_str = converter.decode_beamsearch(preds, beamWidth=2)

        else:
            # ── Attention branch ─────────────────────────────────────────────
            # is_train=False → autoregressive decoding (no teacher forcing):
            #   the decoder uses its own previous prediction as the next input
            #   instead of the ground-truth token.
            preds = model(image, text_for_pred, is_train=False)  # [B, max_len, num_class]
            forward_time = time.time() - start_time

            # Align prediction and target lengths:
            #   preds[:, :-1, :] skips the prediction for the last position
            #   text_for_loss[:, 1:] removes the [GO] start token
            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]),
                             target.contiguous().view(-1))

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            # Also decode the ground-truth indices back to strings for comparison
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # ── Per-sample metrics ────────────────────────────────────────────────
        # Softmax → max probability at each time step → confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)   # [B, T] best prob per step
        confidence_score_list = []

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                # Strip everything from [s] onward — that is the padding
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1  # exact match

            # ICDAR2019 Normalised Edit Distance:
            #   edit_distance counts insertions, deletions, substitutions.
            #   Dividing by max(len(gt), len(pred)) normalises for length.
            #   Subtracting from 1 makes 1 = perfect, 0 = worst.
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # Confidence = cumulative product of per-step max probabilities.
            # This is the joint probability that every character is correct,
            # assuming independence between time steps.
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0    # empty prediction

            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data
