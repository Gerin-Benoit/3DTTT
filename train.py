"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina, Nataliia Molchanova
"""

import argparse
import os
import torch
from torch import nn
from monai.data import decollate_batch
from monai.transforms import Compose, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
import numpy as np
import random
from metrics import dice_metric
from data_load import get_train_dataloader, get_val_dataloader
import wandb
from os.path import join as pjoin
from metrics import *
from model import *

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# trainining
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Specify the initial learning rate')
parser.add_argument('--n_epochs', type=int, default=300,
                    help='Specify the number of epochs to train for')
parser.add_argument('--classic_lr', action='store_true', default=False, help='use the constant lr proposed by the '
                                                                             'original project')
# initialisation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
# data

'''
parser.add_argument('--path_train_data', type=str, required=True, 
                    help='Specify the path to the training data files directory')
parser.add_argument('--path_train_gts', type=str, required=True, 
                    help='Specify the path to the training gts files directory')
parser.add_argument('--path_val_data', type=str, required=True, 
                    help='Specify the path to the validation data files directory')
parser.add_argument('--path_val_gts', type=str, required=True, 
                    help='Specify the path to the validation gts files directory')
'''

parser.add_argument('--data_dir', type=str, default='/dir/GrBM/gerinb/data/shift_dataset',
                    help='Specify the path to the data files directory')

parser.add_argument('--I', nargs='+', default=['FLAIR'], choices=['FLAIR', 'T2', 'T1', 'T1ce', 'PD'])

parser.add_argument('--save_path', type=str, default='/dir/GrBM/gerinb/msseg',
                    help='Specify the path to the save directory')

parser.add_argument('--num_workers', type=int, default=12,
                    help='Number of workers')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--cache_rate', default=1.0, type=float)
# logging
parser.add_argument('--val_interval', type=int, default=5,
                    help='Validation every n-th epochs')
parser.add_argument('--threshold', type=float, default=0.4,
                    help='Probability threshold')

parser.add_argument('--wandb_project', type=str, default='msseg', help='wandb project name')
parser.add_argument('--name', default="idiot without a name", help='Wandb run name')

VAL_AMP = True
roi_size = (96, 96, 96)


def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi_size,
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


post_trans = Compose(
    [AsDiscrete(argmax=True, to_onehot=2)]
)


def main(args):
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    set_determinism(seed=seed_val)

    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    save_dir = f'{args.save_path}/{args.name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    wandb.login()
    wandb.init(project=args.wandb_project, entity='max_and_ben')
    wandb.run.name = f'{args.name}_seed_{args.seed}'

    training_paths = [pjoin(args.data_dir, "msseg")]  # , pjoin(args.data_dir, "best")]
    validation_paths = [pjoin(args.data_dir, "msseg")]  # , pjoin(args.data_dir, "best")]

    '''' Initialize dataloaders '''
    train_paths = [pjoin(tp, "train") for tp in training_paths]
    val_paths = [pjoin(tp, "eval_in") for tp in validation_paths]
    training_gts_path = [pjoin(tp, "train", "gt") for tp in training_paths]
    val_gts_path = [pjoin(tp, "eval_in", "gt") for tp in validation_paths]
    val_bms_path = [pjoin(tp, "eval_in", "fg_mask") for tp in validation_paths]

    train_loader = get_train_dataloader(scan_paths=train_paths,
                                        gts_paths=training_gts_path,
                                        num_workers=args.num_workers,
                                        cache_rate=args.cache_rate,
                                        seed=args.seed,
                                        I=args.I)
    val_loader = get_val_dataloader(scan_paths=val_paths,
                                    gts_paths=val_gts_path,
                                    bm_paths=val_bms_path,
                                    num_workers=args.num_workers,
                                    cache_rate=args.cache_rate,
                                    I=args.I)

    ''' Initialise the model '''

    model = UNet3D(in_channels=len(args.I), num_classes=2).to(device)

    print(model)

    loss_function = DiceLoss(to_onehot_y=True,
                             softmax=True, sigmoid=False,
                             include_background=False)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5 if args.classic_lr else args.learning_rate, weight_decay=0.0005)  # wd = 0.0005
    if args.classic_lr:
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=args.n_epochs)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)

    act = nn.Softmax(dim=1)

    epoch_num = args.n_epochs
    val_interval = args.val_interval
    threshold = args.threshold
    gamma_focal = 2.0
    dice_weight = 0.5
    focal_weight = 1.0

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    best_metric_nDSC, best_metric_epoch_nDSC = -1, -1
    best_metric_DSC, best_metric_epoch_DSC = -1, -1

    epoch_loss_values, metric_values_nDSC, metric_values_DSC = [], [], []

    scaler = torch.cuda.amp.GradScaler()

    ''' Training loop '''
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        epoch_loss_ce = 0
        epoch_loss_dice = 0
        step = 0
        for batch_data in train_loader:
            n_samples = batch_data["image"].size(0)
            for m in range(0, batch_data["image"].size(0), args.batch_size):
                step += args.batch_size
                inputs, labels = (
                    batch_data["image"][m:(m + 2)].to(device),
                    batch_data["label"][m:(m + 2)].type(torch.LongTensor).to(device))

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)

                    # Dice loss
                    loss1 = loss_function(outputs, labels)
                    # Focal loss
                    ce_loss = nn.CrossEntropyLoss(reduction='none')
                    ce = ce_loss(outputs, torch.squeeze(labels, dim=1))
                    pt = torch.exp(-ce)
                    loss2 = (1 - pt) ** gamma_focal * ce
                    loss2 = torch.mean(loss2)
                    loss = dice_weight * loss1 + focal_weight * loss2

                epoch_loss += loss.item()
                epoch_loss_ce += loss2.item()
                epoch_loss_dice += loss1.item()

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                if step % 100 == 0:
                    step_print = int(step / args.batch_size)
                    print(
                        f"{step_print}/{(len(train_loader) * n_samples) // (train_loader.batch_size * args.batch_size)}, train_loss: {loss.item():.4f}")

        epoch_loss /= step_print
        epoch_loss_dice /= step_print
        epoch_loss_ce /= step_print
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()

        wandb.log(
            {'Total Loss/train': epoch_loss, 'Dice Loss/train': epoch_loss_dice, 'Focal Loss/train': epoch_loss_ce,
             'Learning rate': current_lr, },  # 'Dice Metric/train': metric},
            step=epoch)

        ''' Validation '''
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                nDSC_list = []
                for val_data in val_loader:
                    val_inputs, val_labels, val_bms = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                        val_data["brain_mask"].squeeze().cpu().numpy()
                    )

                    val_outputs = inference(val_inputs, model)

                    for_dice_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                    dice_metric(y_pred=for_dice_outputs, y=val_labels)

                    val_outputs = act(val_outputs)[:, 1]
                    val_outputs = torch.where(val_outputs >= threshold, torch.tensor(1.0).to(device),
                                              torch.tensor(0.0).to(device))
                    val_outputs = val_outputs.squeeze().cpu().numpy()
                    # curr_preds = thresholded_output.squeeze().cpu().numpy()[val_bms == 1]
                    # gts = val_labels.squeeze().cpu().numpy()[val_bms == 1]
                    # nDSC = dice_norm_metric(gts, curr_preds)
                    nDSC = dice_norm_metric(val_labels.squeeze().cpu().numpy()[val_bms == 1], val_outputs[val_bms == 1])
                    nDSC_list.append(nDSC)

                torch.cuda.empty_cache()
                del val_inputs, val_labels, val_outputs, val_bms, for_dice_outputs  # , thresholded_output, curr_preds, gts , val_bms
                metric_nDSC = np.mean(nDSC_list)
                metric_DSC = dice_metric.aggregate().item()
                wandb.log({'nDSC Metric/val': metric_nDSC, 'DSC Metric/val': metric_DSC}, step=epoch)
                metric_values_nDSC.append(metric_nDSC)
                metric_values_DSC.append(metric_DSC)

                if metric_nDSC > best_metric_nDSC:
                    best_metric_nDSC = metric_nDSC
                    best_metric_epoch_nDSC = epoch + 1
                    save_path = os.path.join(save_dir, f"best_nDSC_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for nDSC")

                if metric_DSC > best_metric_DSC:
                    best_metric_DSC = metric_DSC
                    best_metric_epoch_DSC = epoch + 1
                    save_path = os.path.join(save_dir, f"best_DSC_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for DSC")

                print(f"current epoch: {epoch + 1} current mean normalized dice: {metric_nDSC:.4f}"
                      f"\nbest mean normalized dice: {best_metric_nDSC:.4f} at epoch: {best_metric_epoch_nDSC}"
                      f"\nbest mean dice: {best_metric_DSC:.4f} at epoch: {best_metric_epoch_DSC}"
                      )


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
