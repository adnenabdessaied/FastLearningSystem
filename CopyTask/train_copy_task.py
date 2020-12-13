import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DNC.util import *
from DNC.dnc import DNC
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as var
import torch as T
#from visdom import Visdom
import argparse
import time
import math
import os
import sys
import getopt
import numpy as np
import warnings
from tensorboardX import SummaryWriter
from copy import deepcopy
from tqdm import tqdm
from CopyTask.config.config_loader import load_ini_file
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("-cfg",
                    "--configuration",
                    help="Path to the .ini config file.",
                    required=True)

parser.add_argument("-m",
                    "--mode",
                    help="train or val",
                    required=True,
                    )
args = vars(parser.parse_args())
cfg  = load_ini_file(args["configuration"])
mode = args["mode"]

gpu_id = cfg["hyperparameters"]["gpu_id"]
if gpu_id >= 0:
    assert torch.cuda.is_available()
    device = torch.device("cuda:" + str(gpu_id))
    print("Using GPU {} | {}".format(gpu_id, torch.cuda.get_device_name(gpu_id)))
elif gpu_id == -1:
    device = torch.device("cpu")
    print("Using the CPU")
else:
    raise NotImplementedError(
        "Device ID {} not recognized. gpu_id = 0, 1, 2 etc. Use -1 for CPU".format(gpu_id))


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def generate_data(batch_size, length, size, device):

    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros(
        (batch_size, 2 * length + 1, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, :size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1:, :size - 1] = sequence

    input_data = T.from_numpy(input_data)
    target_output = T.from_numpy(target_output)
    input_data = input_data.to(device)
    target_output = target_output.to(device)

    return var(input_data), var(target_output)


def criterion(predictions, targets):
    return T.mean(
        -1 * F.logsigmoid(predictions) * (targets) - T.log(1 -
                                                           F.sigmoid(predictions) + 1e-9) * (1 - targets)
    )


def accuracy(x, y):
    '''
    x, y: (batch_size, len_sequence, len_vector)

    Return the mean accuracy over batches.
    The accuracy is the percentage of bits correctly generated.
    '''

    with torch.no_grad():
        total_elements = torch.tensor(x.size(1) * x.size(2), dtype=torch.float32)

        accuracy = (torch.sigmoid(x).round() == y).float().sum(dim=1).sum(dim=1) / total_elements

        return torch.mean(accuracy).item()


def _save_checkpoint(name, net, optimizer, step, cfg):
    """Saves the network into a checkpoint.

    Arguments:
        name {str} -- The name of the checkpoint
        net {torch.nn.Module} -- The network we want to checkpoint.
        optimizer {torch.optim.Optimizer} -- The optimizer of the experiment.
        epoch {int} -- The epoch at which the net will be checkpointed.
        batch_iter_tr {int} -- The training batch iteration at checkpointing.
        batch_iter_val {int} -- The validation batch iteration at checkpointing.
        lr {float} -- Learning rate
        cfg {dict} -- Config dict of the experiment.
    """

    torch.save({
        "net"                 : net,
        "net_state_dict"      : net.state_dict(),
        "optimizer"           : optimizer,
        "optimizer_state_dict": optimizer.state_dict(),
        "step"                : step,

    }, os.path.join(
        cfg["logging"]["checkpoints_dir"], name + "_checkpoint.pth"))


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    ckpts_dir = cfg["logging"]["checkpoints_dir"]
    log_dir = cfg["logging"]["tensorboard_dir"]
    tb_writer = SummaryWriter(logdir=log_dir)
    tb_writer_test = SummaryWriter(logdir=log_dir)

    if not os.path.isdir(ckpts_dir):
        os.mkdir(ckpts_dir)

    batch_size = cfg["hyperparameters"]["batch_size"]
    sequence_max_length = cfg["dnc"]["seq_max_length"]
    cfg["hyperparameters"]["num_iterations"]
    # summarize_freq = args.summarize_freq
    # check_freq = args.check_freq

    pass_through_memory = cfg["dnc"]["pass_through_mem"]
    reset = cfg["dnc"]["reset"]
    clip = cfg["hyperparameters"]["clip"]
    rnn = DNC(
            input_size=cfg["dnc"]["input_size"],
            output_size=cfg["dnc"]["output_size"],
            hidden_size=cfg["dnc"]["hidden_dim"],
            rnn_type=cfg["dnc"]["rnn_type"],
            num_layers=cfg["dnc"]["num_layers"],
            num_hidden_layers=cfg["dnc"]["num_layers_hidden"],
            dropout=cfg["hyperparameters"]["dropout"],
            nr_cells=cfg["dnc"]["n"],
            cell_size=cfg["dnc"]["w"],
            read_heads=cfg["dnc"]["r"],
            gpu_id=cfg["hyperparameters"]["gpu_id"],
            debug=cfg["dnc"]["debug"],
            batch_first=True,
            independent_linears=True)
    # register_nan_checks(rnn)

    rnn = rnn.to(device)
    rnn_lstm = deepcopy(rnn)
    rnn_lstm.debug = False

    last_save_losses = []

    if cfg["hyperparameters"]["optimizer"] == 'adam':
        optimizer = optim.Adam(rnn.parameters(), lr=cfg["hyperparameters"]["lr"],
                               eps=1e-9, betas=[0.9, 0.98])  # 0.0001
    elif cfg["hyperparameters"]["optimizer"] == 'adamax':
        optimizer = optim.Adamax(
            rnn.parameters(), lr=cfg["hyperparameters"]["lr"], eps=1e-9, betas=[0.9, 0.98])  # 0.0001
    elif cfg["hyperparameters"]["optimizer"] == 'rmsprop':
        optimizer = optim.RMSprop(
            rnn.parameters(), lr=cfg["hyperparameters"]["lr"], momentum=cfg["hyperparameters"]["momentum"], eps=1e-10)  # 0.0001
    elif cfg["hyperparameters"]["optimizer"] == 'sgd':
        optimizer = optim.SGD(rnn.parameters(), lr=cfg["hyperparameters"]["lr"])  # 0.01
    elif cfg["hyperparameters"]["optimizer"] == 'adagrad':
        optimizer = optim.Adagrad(rnn.parameters(), lr=cfg["hyperparameters"]["lr"])
    elif cfg["hyperparameters"]["optimizer"] == 'adadelta':
        optimizer = optim.Adadelta(rnn.parameters(), lr=cfg["hyperparameters"]["lr"])

    optimizer_lstm = optim.RMSprop(
        rnn_lstm.parameters(), lr=cfg["hyperparameters"]["lr"], momentum=cfg["hyperparameters"]["momentum"], eps=1e-10)  # 0.0001
    (chx, mhx, rv) = (None, None, None)
    (chx_lstm, mhx_lstm, rv_lstm) = (None, None, None)
    step = 0
    step_test = 0
    for sequence_max_length in cfg["dnc"]["seq_max_length"]:
        for epoch in tqdm(range(int(cfg["hyperparameters"]["num_iterations"] / batch_size))):
            optimizer.zero_grad()
            if len(cfg["dnc"]["seq_max_length"]) > 1:
                random_length = sequence_max_length
            else:
                random_length = np.random.randint(2, sequence_max_length + 1)
            random_length = 3
            input_data, target_output = generate_data(
                batch_size, random_length, cfg["dnc"]["input_size"], device)
            if mode == "train":
                rnn.train()
                rnn_lstm.train()
            else:
                rnn.eval()
                rnn_lstm.eval()
            with torch.set_grad_enabled(mode == "train"):
                if rnn.debug:
                    output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None),
                                                    reset_experience=reset, pass_through_memory=pass_through_memory)
                else:
                    output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None),
                                                reset_experience=reset, pass_through_memory=pass_through_memory)

                if rnn_lstm.debug:
                    output_lstm, (chx_lstm, mhx_lstm, rv_lstm), v_lstm = rnn_lstm(input_data, (None, mhx_lstm, None),
                                                    reset_experience=reset, pass_through_memory=False)
                else:
                    output_lstm, (chx_lstm, mhx_lstm, rv_lstm) = rnn_lstm(input_data, (None, mhx_lstm, None),
                                                reset_experience=reset, pass_through_memory=False)

            loss = criterion((output), target_output)
            loss_lstm = criterion((output_lstm), target_output)
            if mode == "train":
                loss.backward()
                T.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
                optimizer.step()

                loss_lstm.backward()
                T.nn.utils.clip_grad_norm_(rnn_lstm.parameters(), clip)
                optimizer_lstm.step()

            loss_value = loss.item()
            loss_lstm_value = loss_lstm.item()

            loss_dict = {"loss_dnc" : loss_value,
                         "loss_lstm": loss_lstm_value}

            tb_writer.add_scalars("Loss", loss_dict, global_step=step)
            # f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
            # ax1.imshow(np.squeeze(output.cpu().numpy(), 0))
            # ax2.imshow(np.squeeze(target_output.cpu().numpy(), 0))
            # ax1.grid(b=False)
            # ax2.grid(b=False)
            # plt.show()
            output_masked = output[:, random_length+1:, :]
            output_lstm_masked = output_lstm[:, random_length+1:, :]
            target = target_output[:, random_length+1:, :]
            acc = accuracy(output_masked, target)
            acc_lstm = accuracy(output_lstm_masked, target)
            acc_dict = {"acc_dnc"  : acc,
                        "acc_lstm" : acc_lstm}
            tb_writer.add_scalars("Accuracy", acc_dict, global_step=step)
            if cfg["dnc"]["debug"]:
                contrib_dict = {
                    "Memory": v["memory_influence"],
                    "Controller": v["controller_influence"]
                }
                tb_writer.add_scalars("Contributions", contrib_dict, global_step=step)
            output_masked = torch.sigmoid(output_masked).round()
            output_lstm_masked = torch.sigmoid(output_lstm_masked).round()
            tb_writer.add_image("Target", target[0, :, :].unsqueeze(-1), global_step=step, dataformats="HWC")
            tb_writer.add_image("Prediction_dnc", output_masked[0, :, :].unsqueeze(-1), global_step=step, dataformats="HWC")
            tb_writer.add_image("Prediction_lstm", output_lstm_masked[0, :, :].unsqueeze(-1), global_step=step, dataformats="HWC")

            # detach memory from graph
            mhx = {k: (v.detach() if isinstance(v, var) else v)
                for k, v in mhx.items()}

            mhx_lstm = {k: (v.detach() if isinstance(v, var) else v)
                for k, v in mhx_lstm.items()}
            step += 1
    _save_checkpoint("dnc_bla", rnn, optimizer, step, cfg)
    _save_checkpoint("lstm_bla", rnn_lstm, optimizer, step, cfg)
print("[INFO] Training Done")
