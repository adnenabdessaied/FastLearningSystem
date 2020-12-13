__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


from data_loader import VqaDataset
import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import logging
import json
import torch
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
from basic_vqa.config.config_loader_advanced import load_ini_file
from torch.utils.data import DataLoader
import torchvision.transforms as T
logging.basicConfig(level=logging.INFO)


def _set_seed(seed):
    """Sets the seed for reproducibility. 

    Args:
        seed (int): The seed used
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


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

    start_epoch     = chkpt["epoch"]
    batch_iter_tr   = chkpt["batch_iter_tr"]
    batch_iter_val  = chkpt["batch_iter_val"]
    lr              = chkpt["lr"]
    epoch           = chkpt["epoch"] + 1
    logging.info("Network loaded from the checkpoint of epoch {}".format(epoch - 1))
    return net, optimizer, start_epoch, batch_iter_tr, batch_iter_val, lr


def test(name, cfg, path_ckpt):
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

    # Load the trained networks
    model,  _, _, _, _, _ = _load_checkpoint(path_ckpt, gpu_id)

    # Send to gpu and switch to eval mode
    model.eval()

    # Load the test dataset
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))])
    dataset = VqaDataset(
        input_dir=cfg["paths"]["input"],
        input_vqa="test.npy",
        max_qst_length=cfg["hyperparameters"]["max_input_length"],
        max_num_ans=cfg["hyperparameters"]["max_num_answers"],
        transform=transform
    )
    N = [256, 512, 1024]
    model.cfg = cfg

    for n in N:
        memory_contributions_q = []
        controller_contributions_q = []
        memory_contributions_c = []
        controller_contributions_c = [] 
        if name == "DNC_Q":
            model.qst_encoder.dnc_q.nr_cells = n
            # model.qst_encoder.dnc_q.debug = False
            model.qst_encoder.cfg = cfg
            model.qst_encoder.dnc_q.controller_contribution = nn.Linear(
                model.qst_encoder.dnc_q.output_size, model.qst_encoder.dnc_q.output_size_final)
            model.qst_encoder.dnc_q.memory_contribution = nn.Linear(
                model.qst_encoder.dnc_q.read_vectors_size, model.qst_encoder.dnc_q.output_size_final)
            (chx_q, mhx_q, rv_q) = (None, None, None)

        elif name == "DNC_C":
            (chx_c, mhx_c, rv_c) = (None, None, None)
            # model.dnc.debug = False
            model.dnc.nr_cells = n
            model.dnc.controller_contribution = nn.Linear(
                model.dnc.output_size, model.dnc.output_size_final)
            model.dnc.memory_contribution = nn.Linear(
                model.dnc.read_vectors_size, model.dnc.output_size_final)
            # (chx_q, mhx_q, rv_q) = (None, None, None)

        elif name == "DNC_QC":
            (chx_c, mhx_c, rv_c) = (None, None, None)
            (chx_q, mhx_q, rv_q) = (None, None, None)
            model.qst_encoder.dnc_q.nr_cells = n
            model.dnc.nr_cells = n
            model.qst_encoder.dnc_q.controller_contribution = nn.Linear(
                model.qst_encoder.dnc_q.output_size, model.qst_encoder.dnc_q.output_size_final)
            model.qst_encoder.dnc_q.memory_contribution = nn.Linear(
                model.qst_encoder.dnc_q.read_vectors_size, model.qst_encoder.dnc_q.output_size_final)
            model.dnc.controller_contribution = nn.Linear(
                model.dnc.output_size, model.dnc.output_size_final)
            model.dnc.memory_contribution = nn.Linear(
                model.dnc.read_vectors_size, model.dnc.output_size_final)

        model.to(device)
        _set_seed(cfg["hyperparameters"]["seed"])
        dataloader = DataLoader(dataset, batch_size=cfg["hyperparameters"]["batch_size"], shuffle=False)
        ans_list = dataset.ans_vocab.word_list
        pbar = tqdm(dataloader, desc="Testing")
        predictions = []
        # if cfg["hyperparameters"]["use_dnc_c"]:
        #     if cfg["dnc_c"]["type"] == "MLP":
        #         (mhx_c, rv_c) = model.dnc.init_hidden(None, cfg["hyperparameters"]["batch_size"], False)
        #     else:
        #         (chx_c, mhx_c, rv_c) = (None, None, None)

        # if cfg["hyperparameters"]["use_dnc_q"]:
        #     (chx_q, mhx_q, rv_q) = (None, None, None)

        for batch_idx, batch_sample in enumerate(pbar):
            image = batch_sample['image'].to(device)
            question = batch_sample['question'].to(device)
            question_ids = batch_sample["question_id"].tolist()

            if image.size(0) != cfg["hyperparameters"]["batch_size"]:
                if cfg["hyperparameters"]["use_dnc_c"]:
                    if cfg["dnc_c"]["type"] == "MLP":
                        (mhx_c, rv_c) = model.dnc.init_hidden(None, image.size(0), False)
                    else:
                        (chx_c, mhx_c, rv_c) = (None, None, None)

                if cfg["hyperparameters"]["use_dnc_q"]:
                    (chx_q, mhx_q, rv_q) = (None, None, None)

            with torch.no_grad():
                # if cfg["hyperparameters"]["use_dnc_q"] and cfg["hyperparameters"]["use_dnc_c"]:
                if name == "DNC_QC":
                    if cfg["dnc_c"]["type"] == "MLP":
                        output, (chx_q, mhx_q, rv_q), v_q, (mhx_c, rv_c), v_c = model(
                            image, question, chx_q=chx_q, mhx_q=mhx_q, rv_q=rv_q, mhx_c=mhx_c, rv_c=rv_c)
                    else:
                        output, (chx_q, mhx_q, rv_q), v_q, (chx_c, mhx_c, rv_c), v_c = model(
                            image, question, chx_q=chx_q, mhx_q=mhx_q, rv_q=rv_q, chx_c=chx_c, mhx_c=mhx_c, rv_c=rv_c)
                        memory_contributions_q.append(v_q["memory_influence"].item())
                        controller_contributions_q.append(v_q["controller_influence"].item())
                        memory_contributions_c.append(v_c["memory_influence"].item())
                        controller_contributions_c.append(v_c["controller_influence"].item())
                # elif cfg["hyperparameters"]["use_dnc_q"] and not cfg["hyperparameters"]["use_dnc_c"]:
                elif name == "DNC_Q":
                    output, (chx_q, mhx_q, rv_q), v_q = model(image, question, chx=chx_q, mhx=mhx_q, rv=rv_q)
                    memory_contributions_q.append(v_q["memory_influence"].item())
                    controller_contributions_q.append(v_q["controller_influence"].item())
                # elif not cfg["hyperparameters"]["use_dnc_q"] and cfg["hyperparameters"]["use_dnc_c"]:
                elif name == "DNC_C":
                    if cfg["dnc_c"]["type"] == "MLP":
                        output, (mhx_c, rv_c), v_c = model(
                            image, question, mhx_c=mhx_c, rv_c=rv_c)
                    else:
                        output, (chx_c, mhx_c, rv_c), v_c = model(
                            image, question, chx_c=chx_c, mhx_c=mhx_c, rv_c=rv_c)
                        memory_contributions_c.append(v_c["memory_influence"].item())
                        controller_contributions_c.append(v_c["controller_influence"].item())
                else:
                    output = model(image, question)
            _, pred = torch.max(output, 1)  # [batch_size]
            pred = pred.tolist()
            pred = [ans_list[i] for i in pred]
            for id_, ans in zip(question_ids, pred):
                predictions.append({
                    "question_id": id_,
                    "answer": ans
                })

        testFile = os.path.join(cfg["logging"]["results_dir"], name+"_" + n + "_test_res.json")
        with open(testFile, 'w') as f:
            json.dump(predictions, f)
        with open(os.path.join(cfg["logging"]["results_dir"], "memory_contrib_q_" + name + "_" + str(n) + "_.pkl"), "wb") as f:
            pickle.dump(memory_contributions_q, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(cfg["logging"]["results_dir"], "controller_contrib_q_" + name + "_" + str(n) + "_.pkl"), "wb") as f:
            pickle.dump(controller_contributions_q, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(cfg["logging"]["results_dir"], "memory_contrib_c_" + name + "_" + str(n) + "_.pkl"), "wb") as f:
            pickle.dump(memory_contributions_c, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(cfg["logging"]["results_dir"], "controller_contrib_c_" + name + "_" + str(n) + "_.pkl"), "wb") as f:
            pickle.dump(controller_contributions_c, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-n",
                            "--name",
                            required=True,
                            help="Name of the json file that will hold the results.")

    arg_parser.add_argument("-cfg",
                            "--configuration",
                            required=True,
                            help="Path to the configuration ini file")

    arg_parser.add_argument("-ckpt",
                            "--checkpoint",
                            required=True,
                            help="Path to the checkpointed model.")

    args = vars(arg_parser.parse_args())
    cfg = load_ini_file(args["configuration"])
    path_ckpt = args["checkpoint"]
    name = args["name"]
    test(name, cfg, path_ckpt)
