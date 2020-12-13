__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from tqdm import tqdm
from bAbI.config.config_loader import load_ini_file
from bAbI.nets import bAbI_dnc, bAbI_lstm
from bAbI.preprocessing.babi_dataset import bAbI_Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable as var
from utils import (
    MaskedCrossEntropy,
    accuracy,
    get_tb_text_babi,
    get_predictions,
    regroup,
    _save_checkpoint,
    _set_seed)
import logging
import pickle
from copy import deepcopy
logging.basicConfig(level=logging.INFO)


def train(cfg):
    gpu_id = cfg["hyperparameters"]["gpu_id"]
    # Use GPU if available
    if gpu_id >= 0:
        assert torch.cuda.is_available()
        device = torch.device("cuda:" + str(gpu_id))
        logging.info("Using GPU {} | {}".format(gpu_id, torch.cuda.get_device_name(gpu_id)))
    elif gpu_id == -1:
        device = torch.device("cpu")
        logging.info("Using the CPU")
    else:
        raise NotImplementedError(
            "Device ID {} not recognized. gpu_id = 0, 1, 2 etc. Use -1 for CPU".format(gpu_id))
    _set_seed(cfg["hyperparameters"]["seed"])
    net = bAbI_dnc(cfg)
    _set_seed(cfg["hyperparameters"]["seed"])
    net_lstm = bAbI_lstm(cfg)

    if cfg["hyperparameters"]["distributed_tr"]:
        net      = nn.DataParallel(net).to(device)
        net_lstm = nn.DataParallel(net_lstm).to(device)
    else:
        net      = net.to(device)
        net_lstm = net_lstm.to(device)
    logging.info("Network successfully constructed")
    optimizer = optim.RMSprop(net.parameters(), lr=cfg["hyperparameters"]["lr"], momentum=cfg["hyperparameters"]["momentum"], eps=1e-10)
    optimizer_lstm = optim.RMSprop(
        net_lstm.parameters(), lr=cfg["hyperparameters"]["lr"], momentum=cfg["hyperparameters"]["momentum"], eps=1e-10)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8)
    start_epoch     = 0
    batch_iter_tr   = 0
    batch_iter_val  = 0

    # Prepare train and validation datasets
    logging.info("Loading the data ...")
    tr_dataset = bAbI_Dataset(
        cfg["paths"]["train_data"],
        cfg["paths"]["lexicon"],
        cfg["hyperparameters"]["use_one_hot"]
        )
    val_dataset = bAbI_Dataset(
        cfg["paths"]["val_data"],
        cfg["paths"]["lexicon"],
        cfg["hyperparameters"]["use_one_hot"])

    # Prepare the dataloaders
    tr_dataloader = DataLoader(tr_dataset,
                               batch_size=cfg["hyperparameters"]["batch_size"],
                               shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg["hyperparameters"]["batch_size"],
                                shuffle=True)
    loss_weight = tr_dataset.loss_weight.to(device)
    num_classes = tr_dataset.len_lexicon
    modes       = ["TRAIN", "VAL"]
    datasets    = {"TRAIN": tr_dataset,
                   "VAL"  : val_dataset}
    dataloaders = {"TRAIN": tr_dataloader,
                   "VAL"  : val_dataloader}

    # Define tensorBoard logger
    summary_writer    = SummaryWriter(cfg["logging"]["tensorboard_dir"])
    idx_to_word       = pickle.load(open(cfg["paths"]["idx_to_word"], "rb"))
    num_epochs        = cfg["hyperparameters"]["num_epochs"]
    lr                = 0
    (chx, mhx, rv) = (None, None, None)
    for epoch in range(start_epoch, num_epochs):
        for mode in modes:
            if mode == "TRAIN":
                net.train()
                net_lstm.train()
            else:
                net.eval()
                net_lstm.eval()
            loss_epoch = 0
            acc_epoch  = 0
            loss_epoch_lstm = 0
            acc_epoch_lstm  = 0
            dataset = datasets[mode]
            dataloader = dataloaders[mode]
            prog_bar   = tqdm(dataloader)
            prog_bar.set_description("{} | Epoch {}/{}".format(mode, epoch, num_epochs))
            for batch in prog_bar:
                # Send the data to the appropriate device
                for k in batch.keys():
                    if k in ["input", "output", "mask_answer"]:
                        batch[k] = batch[k].to(device)

                optimizer.zero_grad()
                optimizer_lstm.zero_grad()
                with torch.set_grad_enabled(mode == "TRAIN"):
                    if batch["input"].size(0) != cfg["hyperparameters"]["batch_size"]:
                        continue
                    if cfg["dnc"]["debug"]:
                        logits, (chx, mhx, rv), v = net(batch["input"],
                                                        None, mhx, None,
                                                        reset_experience=cfg["dnc"]["reset_experience"],
                                                        pass_through_memory=cfg["dnc"]["pass_through_mem"])
                    else:
                        logits, (chx, mhx, rv) = net(batch["input"],
                                                     None, mhx, None,
                                                     reset_experience=cfg["dnc"]["reset_experience"],
                                                     pass_through_memory=cfg["dnc"]["pass_through_mem"])

                    logits_lstm = net_lstm(batch["input"])
                    logits_ = logits.view(-1, num_classes)
                    logits_lstm_ = logits_lstm.view(-1, num_classes)
                    output = batch["output"].view(-1)
                    mask = batch["mask_answer"].view(-1)
                    loss = MaskedCrossEntropy(logits_, output, mask)
                    loss_lstm = MaskedCrossEntropy(logits_lstm_, output, mask)

                    if mode == "TRAIN":
                        # Back propagation with anomaly detection -> Makes it easier to locate the faulty parts of the net
                        # if some undesirable phenomena happen, e.g. if some layers produce NaN of Inf values.
                        # with torch.autograd.detect_anomaly():
                        loss.backward()
                        loss_lstm.backward()
                        # clip gradients to avoid explosion
                        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                        nn.utils.clip_grad_norm_(net_lstm.parameters(), 10.0)

                        optimizer.step()
                        optimizer_lstm.step()
                        lr = optimizer.param_groups[0]["lr"]
                        summary_writer.add_scalar("lr_DNC", lr, global_step=batch_iter_tr)

                mhx = {k: (v.detach() if isinstance(v, var) else v) for k, v in mhx.items()}
                loss_dict = {"DNC"   : loss.item(),
                             "LSTM"  : loss_lstm.item()
                             }
                summary_writer.add_scalars("Loss_" + mode + "_Batch",
                                           loss_dict,
                                           global_step=batch_iter_tr if mode == "TRAIN" else batch_iter_val)
                # accumulate batch losses
                loss_epoch += loss.item()
                loss_epoch_lstm += loss_lstm.item()
                answers_gt = regroup(batch["answers_gt"], batch["length_answers"])

                predictions_dnc = get_predictions(logits, batch["answers_idx"])
                predictions_dnc = regroup(predictions_dnc, batch["length_answers"])
                acc, acc_list   = accuracy(answers_gt, predictions_dnc)

                predictions_lstm = get_predictions(logits_lstm, batch["answers_idx"])
                predictions_lstm = regroup(predictions_lstm, batch["length_answers"])
                acc_lstm, acc_list_lstm = accuracy(answers_gt, predictions_lstm)


                acc_dict = {"DNC"   : acc,
                            "LSTM"  : acc_lstm
                            }
                tb_text_dnc    = get_tb_text_babi(answers_gt, predictions_dnc, acc_list, idx_to_word, 8)
                tb_text_lstm   = get_tb_text_babi(answers_gt, predictions_lstm, acc_list_lstm, idx_to_word, 8)
                summary_writer.add_scalars("Accuracy_" + mode + "_Batch",
                                        acc_dict,
                                        global_step=batch_iter_tr if mode == "TRAIN" else batch_iter_val)

                summary_writer.add_text("Output_DNC_" + mode + "_Batch",
                                        tb_text_dnc,
                                        global_step=batch_iter_tr if mode == "TRAIN" else batch_iter_val)

                summary_writer.add_text("Output_LSTM_" + mode + "_Batch",
                                        tb_text_lstm,
                                        global_step=batch_iter_tr if mode == "TRAIN" else batch_iter_val)

                # Accumulate accuracy
                acc_epoch  += acc
                acc_epoch_lstm += acc_lstm

                # update the learning rate using the cyclic scheme
                # and log in into tensorboard.
                # if mode == "TRAIN":
                #     if cfg["hyperparameters"]["lr_schedule"] == "cyclic":
                #         lr = scheduler.get_lr()[0]
                #         scheduler.step()
                #     else:
                #         lr = optimizer.param_groups[0]["lr"]
                #     summary_writer.add_scalar("lr", lr, global_step=batch_iter_tr)                  

                # Release GPU memory cache
                torch.cuda.empty_cache()
                if mode == "TRAIN":
                    batch_iter_tr  += 1
                else:
                    batch_iter_val += 1

            # Compute average epoch and loss accuracies
            if len(dataset) % cfg["hyperparameters"]["batch_size"] == 0:
                total = len(dataloader)
            else:
                total = len(dataloader) - 1
            loss_epoch          /= total
            loss_epoch_lstm     /= total
            acc_epoch           /= total
            acc_epoch_lstm      /= total

            loss_dict = {"DNC": loss_epoch, "LSTM": loss_epoch_lstm}
            acc_dict  = {"DNC": acc_epoch, "LSTM": acc_epoch_lstm}

            summary_writer.add_scalars("Loss_" + mode + "_Epoch", loss_dict, global_step=epoch)
            summary_writer.add_scalars("Accuracy_" + mode + "_Epoch", acc_dict, global_step=epoch)

            if mode == "TRAIN":
                _save_checkpoint("dnc", net, optimizer, epoch, batch_iter_tr, batch_iter_val, lr, cfg)
                _save_checkpoint("lstm", net_lstm, optimizer_lstm, epoch, batch_iter_tr, batch_iter_val, lr, cfg)
                lr_scheduler.step()
                # Delete the oldest checkpoint if the number of checkpoints exceeds 10 to save disk space.
                checkpoints = [ckpt for ckpt in os.listdir(cfg["logging"]["checkpoints_dir"]) if ckpt.endswith("pth")]
                checkpoints = [os.path.join(cfg["logging"]["checkpoints_dir"], checkpoint) for checkpoint in checkpoints]
                if len(checkpoints) > 10:
                    oldest_checkpoint_pth = min(
                        checkpoints, key=lambda s: int(s.split("/")[-1].split("_")[-1].split(".")[0]))
                    os.remove(oldest_checkpoint_pth)
                    if "dnc" in oldest_checkpoint_pth:
                        oldest_checkpoint_pth = oldest_checkpoint_pth.replace("dnc", "lstm")
                        os.remove(oldest_checkpoint_pth)
                    elif "lstm" in oldest_checkpoint_pth:
                        oldest_checkpoint_pth = oldest_checkpoint_pth.replace("lstm", "dnc")
                        os.remove(oldest_checkpoint_pth)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-cfg",
                            "--configuration",
                            required=True,
                            help="Path to the configuration ini file")
    args = vars(arg_parser.parse_args())
    cfg = load_ini_file(args["configuration"])
    train(cfg)
