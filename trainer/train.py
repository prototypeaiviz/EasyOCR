import os
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler   # mixed-precision training
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    # Prints every trainable parameter's name and size, then returns the total.
    # Useful after freezing layers to verify which parts are actually being trained.
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train(opt, show_number = 2, amp=False):
    # ─────────────────────────────────────────────────────────────────────────
    # DATASET SETUP
    # ─────────────────────────────────────────────────────────────────────────
    # opt.select_data: dash-separated list of dataset folder names, e.g. "MJ-ST"
    # opt.batch_ratio: matching dash-separated fractions,         e.g. "0.5-0.5"
    # Each fraction controls how many samples from that dataset appear per batch.
    # This "balanced batching" prevents a large dataset from dominating training.
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')

    # Batch_Balanced_Dataset creates one DataLoader per dataset folder, then
    # get_batch() pulls the correct number of samples from each loader every step.
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")

    # Validation loader: fixed batch of up to 32, shuffled so we see a random
    # slice each evaluation interval (not always the same easy samples).
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True,
        num_workers=int(opt.workers), prefetch_factor=512,
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    # CTCLabelConverter: blank token at index 0, characters at 1..N
    # AttnLabelConverter: adds [GO] (start) and [s] (end) tokens for seq2seq
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)  # vocab size including special tokens

    if opt.rgb:
        opt.input_channel = 3   # RGB input instead of grayscale
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # ── Loading pretrained weights (for fine-tuning) ──────────────────────────
    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model)

        if opt.new_prediction:
            # CASE: fine-tuning on a new character set with a DIFFERENT vocab size.
            # The pretrained Prediction head has shape [hidden → old_num_class].
            # We temporarily resize it to old_num_class so load_state_dict won't
            # complain about mismatched shapes, then replace it after loading.
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))

        model = torch.nn.DataParallel(model).to(device)
        print(f'loading pretrained model from {opt.saved_model}')

        if opt.FT:
            # strict=False: silently ignores keys that are missing or unexpected.
            # Use this when fine-tuning with a different vocab or when you've
            # added/removed layers (e.g. new_prediction).
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            # strict=True (default): every key must match exactly.
            # Use when continuing training on the SAME task/vocab.
            model.load_state_dict(pretrained_dict)

        if opt.new_prediction:
            # Replace the old Prediction head with a fresh one sized for the NEW vocab.
            # Kaiming init is suited for layers followed by ReLU; for a linear
            # output layer it is slightly aggressive but works in practice.
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device)
    else:
        # ── Training from scratch: initialise all weights ─────────────────────
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                # TPS localisation network fc2 has a carefully hand-crafted init
                # (control points laid out on a regular grid); skip it.
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:
                # BatchNorm weight is a 1-D scalar; kaiming_normal_ needs ≥2D.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)

    model.train()
    print("Model:")
    print(model)
    count_parameters(model)

    # ─────────────────────────────────────────────────────────────────────────
    # LOSS FUNCTION
    # ─────────────────────────────────────────────────────────────────────────
    if 'CTC' in opt.Prediction:
        # CTCLoss:
        #   - Input: log-softmax probabilities  [T, B, C]  (time-first format)
        #   - Targets: flat concatenated label indices + lengths
        #   - zero_infinity=True: replaces inf/nan losses with 0 (happens when
        #     sequence is shorter than the number of distinct target characters,
        #     which can occur on very short crops early in training).
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        # CrossEntropyLoss for attention-based decoder:
        #   - Works token-by-token (teacher-forced): at each step the loss is
        #     measured between the predicted token and the next ground-truth token.
        #   - ignore_index=0: ignores the [GO] padding token in the loss.
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    # Running average of the training loss, reset each valInterval steps.
    loss_avg = Averager()

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER FREEZING  (fine-tuning strategy)
    # ─────────────────────────────────────────────────────────────────────────
    # When fine-tuning on a new dataset/language you often want to keep the
    # pretrained CNN features fixed (they generalise well) and only update the
    # LSTM and prediction head.  Freezing is done by setting requires_grad=False
    # on those parameters so they are excluded from optimizer updates.
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass

    # Build the list of parameters the optimizer will actually update.
    # filter(...requires_grad) skips frozen layers automatically.
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMIZER
    # ─────────────────────────────────────────────────────────────────────────
    # Adadelta (default): adaptive per-parameter learning rate, no manual lr decay needed.
    # Adam: faster convergence but sometimes less stable; useful for fine-tuning.
    if opt.optim=='adam':
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    # Save the full configuration to a text file for reproducibility
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    # EasyOCR trains by iteration count (not epochs) because Batch_Balanced_Dataset
    # wraps around indefinitely.  This lets you mix datasets of different sizes
    # with precise ratio control.

    start_iter = 0
    if opt.saved_model != '':
        # Resume from a checkpoint: parse the iteration number from the filename
        # (e.g. "iter_10000.pth" → start at 10000).
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    # GradScaler is used only with AMP (automatic mixed precision).
    # It scales the loss up before backward() to prevent fp16 underflow,
    # then unscales gradients before the optimizer step.
    scaler = GradScaler()
    t1= time.time()

    while(True):
        # zero_grad(set_to_none=True) is faster than zero_grad() because it
        # releases the gradient memory entirely instead of filling with zeros.
        optimizer.zero_grad(set_to_none=True)

        if amp:
            # ── AMP training path ─────────────────────────────────────────────
            # autocast() runs the forward pass in fp16 where safe (Conv, Linear)
            # and keeps fp32 where precision matters (BatchNorm, loss).
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                # converter.encode: turns ['cat','dog'] → flat index tensor + lengths
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    # CTCLoss expects [T, B, C] not [B, T, C]
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)              # [T, B, C]
                    # cudnn's CTC implementation has non-determinism issues with
                    # some versions; disable it around the loss call as a workaround.
                    torch.backends.cudnn.enabled = False
                    cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    torch.backends.cudnn.enabled = True
                else:
                    # Attention: teacher-forcing — feed ground-truth prefix at each step.
                    # text[:, :-1]: all tokens except the last (input to decoder)
                    # text[:, 1:]:  all tokens except [GO] (prediction targets)
                    preds = model(image, text[:, :-1])
                    target = text[:, 1:]
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # scaler.scale multiplies the loss by the current scale factor before backward
            scaler.scale(cost).backward()
            # unscale_ converts scaled gradients back to true values for clipping
            scaler.unscale_(optimizer)
            # Gradient clipping prevents exploding gradients in the LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)      # applies updates (skips if grads have inf/nan)
            scaler.update()             # adjusts scale factor for next iteration
        else:
            # ── Standard fp32 training path ──────────────────────────────────
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])
                target = text[:, 1:]
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        loss_avg.add(cost)

        # ── Validation & checkpoint saving ────────────────────────────────────
        if (i % opt.valInterval == 0) and (i!=0):
            print('training time: ', time.time()-t1)
            t1=time.time()
            elapsed_time = time.time() - start_time

            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
                model.train()

                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                # Save the best model by exact-match accuracy AND by normalised edit distance.
                # Norm-ED is more informative for scripts where full-word matches are rare.
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # Print a few random (gt, predicted, confidence) rows from the validation batch
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'

                start = random.randint(0,len(labels) - show_number )
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                print('validation time: ', time.time()-t1)
                t1=time.time()

        # Periodic checkpoint every 10 000 iterations (allows resuming training)
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
        i += 1
