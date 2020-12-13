__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import torch
import torch.nn as nn
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from tqdm import tqdm
from bAbI.config.config_loader import load_ini_file
from bAbI.preprocessing.babi_dataset import bAbI_Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable as var
from utils import regroup, accuracy, get_predictions, get_tb_test_summary
from utils import _load_checkpoint, _set_seed
import pickle
import matplotlib.pylab as plt
import numpy as np
plt.style.use("ggplot")

logging.basicConfig(level=logging.INFO)


def test(cfg):
    gpu_id = cfg["hyperparameters"]["gpu_id"]

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

    # Filter the most recent checkpoints
    checkpoints      = [ckpt for ckpt in os.listdir(cfg["logging"]["checkpoints_dir"]) if ckpt.endswith("pth")]
    checkpoints_dnc  = list(filter(lambda s: "dnc" in s, checkpoints))
    checkpoints_lstm = list(filter(lambda s: "lstm" in s, checkpoints))
    checkpoints_dnc  = [os.path.join(cfg["logging"]["checkpoints_dir"], checkpoint) for checkpoint in checkpoints_dnc]
    checkpoints_lstm = [os.path.join(cfg["logging"]["checkpoints_dir"], checkpoint) for checkpoint in checkpoints_lstm]

    chkpt_dnc  = max(checkpoints_dnc, key=lambda s: int(s.split("/")[-1].split("_")[-1].split(".")[0]))
    chkpt_lstm = max(checkpoints_lstm, key=lambda s: int(s.split("/")[-1].split("_")[-1].split(".")[0]))

    # Load the trained networks
    net_dnc,  _, _, _, _, _ = _load_checkpoint(chkpt_dnc, gpu_id)
    net_lstm, _, _, _, _, _ = _load_checkpoint(chkpt_lstm, gpu_id)

    net_dnc.rnn.controller_contribution = nn.Linear(net_dnc.rnn.output_size, net_dnc.rnn.output_size_final)
    net_dnc.rnn.memory_contribution = nn.Linear(net_dnc.rnn.read_vectors_size, net_dnc.rnn.output_size_final)

    # Send to gpu and switch to eval mode
    net_dnc.to(device)
    net_lstm.to(device)
    net_dnc.eval()
    net_lstm.eval()

    tb_writer = SummaryWriter(log_dir=cfg["logging"]["tensorboard_dir"])
    # Load the test datasets
    test_datasets = [bAbI_Dataset(
        path, cfg["paths"]["lexicon"], cfg["hyperparameters"]["use_one_hot"]) for path in cfg["paths"]["test_data"]]
    accuracy_dnc_list  = []
    accuracy_lstm_list = []
    tasks = []
    for dataset in test_datasets:
        if dataset.task == "qa4":
            (chx, mhx, rv) = (None, None, None)
            _set_seed(cfg["hyperparameters"]["seed"])
            dataloader = DataLoader(dataset, batch_size=cfg["hyperparameters"]["batch_size"])
            prog_bar = tqdm(dataloader)
            prog_bar.set_description("TASK: " + dataset.task)
            acc_dnc_total  = 0.0
            acc_lstm_total = 0.0
            for i, batch in enumerate(prog_bar):
                for k in batch.keys():
                    if k in ["input", "output", "weights"]:
                        batch[k] = batch[k].to(device)
                if batch["input"].size(0) != cfg["hyperparameters"]["batch_size"]:
                    continue
                with torch.no_grad():
                    if cfg["dnc"]["debug"]:
                        logits, (chx, mhx, rv), v = net_dnc(batch["input"],
                                                            None, mhx, None,
                                                            reset_experience=cfg["dnc"]["reset_experience"],
                                                            pass_through_memory=cfg["dnc"]["pass_through_mem"])
                    else:
                        logits, (chx, mhx, rv) = net_dnc(batch["input"],
                                                        None, mhx, None,
                                                        reset_experience=cfg["dnc"]["reset_experience"],
                                                        pass_through_memory=cfg["dnc"]["pass_through_mem"])

                    logits_lstm = net_lstm(batch["input"])
                answers_gt = regroup(batch["answers_gt"], batch["length_answers"])
                predictions_dnc = get_predictions(logits, batch["answers_idx"])
                predictions_dnc = regroup(predictions_dnc, batch["length_answers"])
                acc_dnc, _ = accuracy(answers_gt, predictions_dnc)
                mhx = {k: (v.detach() if isinstance(v, var) else v) for k, v in mhx.items()}
                predictions_lstm = get_predictions(logits_lstm, batch["answers_idx"])
                predictions_lstm = regroup(predictions_lstm, batch["length_answers"])
                acc_lstm, _ = accuracy(answers_gt, predictions_lstm)

                acc_dnc_total  += acc_dnc
                acc_lstm_total += acc_lstm

        if len(dataset) % cfg["hyperparameters"]["batch_size"] == 0:
            acc_dnc_total  /= len(dataloader)
            acc_lstm_total /= len(dataloader)
        else:
            acc_dnc_total  /= len(dataloader) - 1
            acc_lstm_total /= len(dataloader) - 1

        accuracy_dnc_list.append(str(round(acc_dnc_total, 2)) + " %")
        accuracy_lstm_list.append(str(round(acc_lstm_total, 2)) + " %")
        tasks.append(dataset.task)
        tb_text_summary = get_tb_test_summary(accuracy_dnc_list, accuracy_lstm_list, tasks)
        tb_writer.add_text(tasks[-1], tb_text_summary, global_step=0)
    print(tb_text_summary)
    logging.info("Done Testing...")
    plt.show()

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-cfg",
                            "--configuration",
                            # required=True,
                            default="/home/adnen/Documents/Studium/MScThesis/src/bAbI/config/bAbI_reduced.ini",
                            help="Path to the configuration ini file")

    args = vars(arg_parser.parse_args())
    cfg = load_ini_file(args["configuration"])
    test(cfg)
