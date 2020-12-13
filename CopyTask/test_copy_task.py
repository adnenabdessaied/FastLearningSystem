import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DNC.util import *
from DNC.dnc import DNC
import argparse
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable as var
from CopyTask.config.config_loader import load_ini_file
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-cfg",
                    "--configuration",
                    help="Path to the .ini config file.",
                    # required=True,
                    default="/home/adnen/Documents/Studium/MScThesis/src/CopyTask/config/copy_task_no_curriculum.ini")
parser.add_argument("-ckpt",
                    "--checkpoint",
                    help="Path to the .ini config file.",
                    # required=True,
                    default="/data/models/copy/dnc_checkpoint.pth")
args = vars(parser.parse_args())
cfg  = load_ini_file(args["configuration"])
ckpt = args["checkpoint"]

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


def _load_checkpoint(path, gpu_id):
    """Loads a saved checkpoint.

    Arguments:
        checkpoints_path {str} --  Path to the folder containing the checkpoints.
        gpu_id {int} -- The gpu id.
    Returns:
        tuple -- net, optimizer, start_epoch, batch_iter_tr, batch_iter_val
    """
    if gpu_id >= 0:
        chkpt  = torch.load(path, map_location=lambda storage, loc: storage.cuda(gpu_id))
    else:
        chkpt  = torch.load(path)

    net        = chkpt["net"]
    net.load_state_dict(chkpt["net_state_dict"])

    # We want to train further
    net.train()

    # Send the net first to the device to avoid potential runtime errors 
    # caused by the optimizer if we resume training on a different device.

    optimizer       = chkpt["optimizer"]
    optimizer.load_state_dict(chkpt["optimizer_state_dict"])

    print("Network loaded")
    return net, optimizer


def memory_snapshot(memory, free_gates, alloc_gates, write_gates, n, w, output_dir, iter=None):
    memory = memory.mean(0)
    memory = memory.reshape(n, w)

    # Scale to [0, 1]
    if not (memory.min() == 0 and memory.max() == 0):
        memory  = (memory - memory.min()) / (memory.max() - memory.min())
    # convert the data of the matrix to a colormap
    _cmap   = cm.get_cmap("viridis")
    memory  = _cmap(memory)
    free_gates = _cmap(free_gates)
    alloc_gates = _cmap(alloc_gates)
    write_gates = _cmap(write_gates)

    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    axs.imshow(memory)
    axs.set_title("memory @ time {}".format(iter))
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xlabel('W = {}'.format(w))
    axs.set_ylabel('N = {}'.format(n))
    axs[0].imshow(memory)
    axs[0].set_title("memory @ time {}".format(iter))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlabel('W = {}'.format(w))
    axs[0].set_ylabel('N = {}'.format(n))

    fg = axs[1].imshow(free_gates)
    # axs[1].set_title("")
    axs[1].set_ylabel('free gate')
    axs[1].set_yticks([])

    axs[1].set_ylabel('value')

    ag = axs[2].imshow(alloc_gates)
    # axs[2].set_title("allocation gate")
    axs[2].set_ylabel('allocation gate')
    axs[2].set_yticks([])

    axs[2].set_ylabel('value')

    wg = axs[3].imshow(write_gates)
    axs[3].set_title("write gate")
    axs[3].set_ylabel('write gate')
    axs[3].set_xlabel('time')

    axs[3].set_yticks([])

    axs[3].set_ylabel('value')

    plt.imshow(memory)

    plt.title()
    plt.axis('off')
    plt.xlabel("W = {}".format(w))
    plt.ylabel("N = {}".format(n))
    plt.show()
    plt.colorbar(fg, ax=axs[1])
    plt.colorbar(ag, ax=axs[2])
    plt.colorbar(wg, ax=axs[3])
    plt.show()
    fname = os.path.join(output_dir, str(iter) + ".jpeg")
    plt.savefig(fname)


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


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    in_, ou_ = generate_data(2, 3, 6, device)
    ckpts_dir = cfg["logging"]["checkpoints_dir"]
    log_dir = cfg["logging"]["tensorboard_dir"]
    pass_through_memory = cfg["dnc"]["pass_through_mem"]
    reset = cfg["dnc"]["reset"]
    model, _ = _load_checkpoint(ckpt, cfg["hyperparameters"]["gpu_id"])
    n_batch_test = int(100 / cfg["hyperparameters"]["batch_size"])
    acc_test_dnc = 0
    acc_test_lstm = 0
    (chx, mhx, rv) = (None, None, None)
    model.eval()
    free_gates = np.zeros((2, 62))
    allocation_gates = np.zeros((2, 62))
    write_gates = np.zeros((2, 62))
    memory = np.zeros((cfg["hyperparameters"]["batch_size"], cfg["dnc"]["n"] * cfg["dnc"]["w"]))
    memory_snapshot(memory,free_gates, allocation_gates, write_gates, cfg["dnc"]["n"], cfg["dnc"]["w"], output_dir, iter=0)
    for i in tqdm(range(n_batch_test)):
        # Test with seq length 30
        input_data_test, target_output_test = generate_data(
            cfg["hyperparameters"]["batch_size"], cfg["dnc"]["length_test"], cfg["dnc"]["input_size"], device)
        with torch.no_grad():
            if cfg["dnc"]["debug"]:
                output, (chx, mhx, rv), v = model(
                    input_data_test,
                    (None, mhx, None),
                    reset_experience=reset,
                    pass_through_memory=pass_through_memory)
                    
                # memory_snapshot(
                #     v["memory"], free_gates, allocation_gates, write_gates, cfg["dnc"]["n"], cfg["dnc"]["w"], output_dir, iter=i+1)

            else:
                output, (chx, mhx, rv) = model(
                    input_data_test,
                    (None, mhx, None),
                    reset_experience=reset,
                    pass_through_memory=pass_through_memory)

        output_masked = torch.sigmoid(output)[:, cfg["dnc"]["length_test"]+1:, :].round()
        target = target_output_test[:, cfg["dnc"]["length_test"]+1:, :]
    
        acc_test_dnc  += accuracy(output_masked, target)
    acc_test_dnc /= n_batch_test
    acc_test_lstm /= n_batch_test
    print("Test_Accuracy_dnc = {} | Test_Accuracy_lstm = {}".format(acc_test_dnc, acc_test_lstm))
