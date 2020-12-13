import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as var
from torch.optim import lr_scheduler
from data_loader import get_loader
from tensorboardX import SummaryWriter
from config.config_loader_advanced import load_ini_file
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import VqaModelDncQC
from vqa_eval.PythonHelperTools.vqaTools.vqa import VQA
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


def get_gaussian_noise(t, η, γ, size, device):
    σ_t = np.math.sqrt((η/((1 + t) ** γ)))
    print(σ_t)
    n_t = torch.normal(0.0, std=σ_t, size=size)
    n_t = n_t.to(device)
    return n_t

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def _save_checkpoint(name, net, optimizer, epoch, batch_iter_tr, batch_iter_val, lr, cfg):
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
        "epoch"               : epoch,
        "batch_iter_tr"       : batch_iter_tr,
        "batch_iter_val"      : batch_iter_val,
        "lr"                  : lr
    }, os.path.join(
        cfg["logging"]["checkpoints_dir"], name + "_checkpoint_{}.pth".format(
            epoch)))


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


# --> https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/6
def plot_grad_flow(named_parameters, output_dir, name):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    fname = os.path.join(output_dir, name + "png")
    plt.savefig(fname)
# <--


def main(cfg):
    gpu_id = cfg["hyperparameters"]["gpu_id"]
    # Use GPU if available
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

    data_loader = get_loader(
        input_dir=cfg["paths"]["input"],
        input_vqa_train="train.npy",
        input_vqa_valid="valid.npy",
        max_qst_length=cfg["hyperparameters"]["max_input_length"],
        max_num_ans=cfg["hyperparameters"]["max_num_answers"],
        batch_size=cfg["hyperparameters"]["batch_size"],
        num_workers=6)

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_list = data_loader['train'].dataset.ans_vocab.word_list
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx
    cfg["hyperparameters"]["qst_vocab_size"] = qst_vocab_size
    cfg["hyperparameters"]["ans_vocab_size"] = ans_vocab_size

    # assert not(cfg["hyperparameters"]["use_dnc_c"] and cfg["hyperparameters"]["use_dnc_q"])
    _set_seed(cfg["hyperparameters"]["seed"])
    model = VqaModelDncQC(cfg).to(device)
    if cfg["hyperparameters"]["finetune"]:
        model.load_pretrained_weights(cfg["hyperparameters"]["fc_flag"])
        model.check_successul_weight_loading(cfg["hyperparameters"]["fc_flag"])
    net_name = "Advanced_dnc"

    criterion = nn.CrossEntropyLoss()
    dnc_q_params = {"params": model.qst_encoder.dnc_q.parameters(), "lr": cfg["dnc_q"]["lr"]}
    dnc_c_params = {"params": model.dnc.parameters(), "lr": cfg["dnc_c"]["lr"]}
    embed_params = {"params": model.qst_encoder.word2vec.parameters()}
    img_encoder_params = {"params": model.img_encoder.fc.parameters()}
    fc_1_params  = {"params": model.fc_1.parameters()}
    fc_2_params  = {"params": model.fc_2.parameters()}


    if cfg["hyperparameters"]["optimizer"] == "adam":
        optimizer = optim.Adam(
            [dnc_q_params, dnc_c_params, embed_params, img_encoder_params, fc_1_params, fc_2_params],
            lr=cfg["hyperparameters"]["lr"],
            weight_decay=cfg["hyperparameters"]["weight_decay"])
    elif cfg["hyperparameters"]["optimizer"] == "sgd":
        optimizer = optim.SGD(
            [dnc_q_params, dnc_c_params, embed_params, img_encoder_params, fc_1_params, fc_2_params],
            lr=cfg["hyperparameters"]["lr"],
            weight_decay=cfg["hyperparameters"]["weight_decay"])
    elif cfg["hyperparameters"]["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(
            [dnc_q_params, dnc_c_params, embed_params, img_encoder_params, fc_1_params, fc_2_params],
            lr=cfg["hyperparameters"]["lr"],
            weight_decay=cfg["hyperparameters"]["weight_decay"])
    else:
        raise ValueError("<{}> is not supported.".format(cfg["hyperparameters"]["optimizer"]))

    print("Training " + net_name)
    print("Using " + cfg["hyperparameters"]["optimizer"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg["hyperparameters"]["lr_reduce_after"], gamma=cfg["hyperparameters"]["lr_decay_rate"])
    summary_writer = SummaryWriter(logdir=cfg["logging"]["tensorboard_dir"])
    tr_iter = 0
    val_iter = 0
    lr = 0
    lr_dnc_q = 0
    lr_dnc_c = 0

    for epoch in range(cfg["hyperparameters"]["num_epochs"]):
        for phase in ["train", "valid"]:
            (chx_q, mhx_q, rv_q) = (None, None, None)
            if cfg["dnc_c"]["type"] == "MLP":
                (mhx_c, rv_c) = model.dnc.init_hidden(None, cfg["hyperparameters"]["batch_size"], False)
            else:
                (chx_c, mhx_c, rv_c) = (None, None, None)
            running_loss = 0.0
            dataloader = data_loader[phase]
            batch_step_size = len(dataloader.dataset) / cfg["hyperparameters"]["batch_size"]
            if phase == "train":
                model.train()
            else:
                model.eval()
                val_predictions = []
            pbar = tqdm(dataloader)
            pbar.set_description("{} | Epcoh {}/{}".format(phase, epoch, cfg["hyperparameters"]["num_epochs"]))
            for batch_idx, batch_sample in enumerate(pbar):
                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.
                if image.size(0) != cfg["hyperparameters"]["batch_size"]:
                    (chx_q, mhx_q, rv_q) = (None, None, None)
                    if cfg["dnc_c"]["type"] == "MLP":
                        (mhx_c, rv_c) = model.dnc.init_hidden(None, image.size(0), False)
                    else:
                        (chx_c, mhx_c, rv_c) = (None, None, None)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if cfg["dnc_c"]["type"] == "MLP":
                        output, (chx_q, mhx_q, rv_q), v_q, (mhx_c, rv_c), v_c = model(
                            image, question, chx_q=chx_q, mhx_q=mhx_q, rv_q=rv_q, mhx_c=mhx_c, rv_c=rv_c)
                    else:
                        output, (chx_q, mhx_q, rv_q), v_q, (chx_c, mhx_c, rv_c), v_c = model(
                            image, question, chx_q=chx_q, mhx_q=mhx_q, rv_q=rv_q, chx_c=chx_c, mhx_c=mhx_c, rv_c=rv_c)
                    _, pred = torch.max(output, 1)  # [batch_size]
                    # _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                    loss = criterion(output, label)
                    if phase == 'train':
                        loss.backward()
                        if cfg["hyperparameters"]["add_noise"]:
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad += get_gaussian_noise(tr_iter, 1.0, 0.55, p.grad.shape, device)
                        # if iter % cfg["hyperparameters"]["grad_flow_interval"] == 0:
                        #     plot_grad_flow(model.named_parameters(), cfg["hyperparameters"]["grad_flow_dir"], str(tr_iter))
                        if cfg["hyperparameters"]["use_clip_grad"]:
                            nn.utils.clip_grad_norm_(model.parameters(), cfg["hyperparameters"]["clip_value"])
                        optimizer.step()

                        lr_dnc_q = optimizer.param_groups[0]["lr"]
                        lr_dnc_c = optimizer.param_groups[1]["lr"]
                        lr = optimizer.param_groups[2]["lr"]
                        dict_lr = {"DNC_Q": lr_dnc_q,
                                   "DNC_C": lr_dnc_c,
                                   "Rest": lr}
                        summary_writer.add_scalars("lr", dict_lr, global_step=tr_iter)

                    else:
                        question_ids = batch_sample["question_id"].tolist()
                        pred = pred.tolist()
                        pred = [ans_list[i] for i in pred]
                        for id_, ans in zip(question_ids, pred):
                            val_predictions.append({
                                "question_id": id_,
                                "answer": ans
                            })

                    mhx_q = {k: (v.detach() if isinstance(v, var) else v) for k, v in mhx_q.items()}
                    mhx_c = {k: (v.detach() if isinstance(v, var) else v) for k, v in mhx_c.items()}
                    if cfg["dnc_c"]["type"] == "MLP":
                        rv_c = rv_c.detach()

                running_loss += loss.item()
                summary_writer.add_scalar("Loss/" + phase + "_Batch",
                                           loss.item(),
                                           global_step=tr_iter if phase == "train" else val_iter)
                if v_c is not None and cfg["dnc_c"]["type"] == "mlp":
                    inf_dict = {
                        "Memory": v_c["memory_influence"],
                        "Controller": v_c["controller_influence"]
                    }
                    summary_writer.add_scalars(
                        "Influences/" + phase + "_Batch",
                        inf_dict,
                        global_step=tr_iter if phase == "train" else val_iter)
                if phase == "train":
                    tr_iter += 1
                else:
                    val_iter += 1
            if phase == "train":
                scheduler.step()

            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            summary_writer.add_scalar("Loss/" + phase + "_Epoch", epoch_loss, global_step=epoch)
            if phase == "valid":
                valFile = os.path.join(cfg["logging"]["results_dir"], "val_res.json")
                with open(valFile, 'w') as f:
                    json.dump(val_predictions, f)
                annFile = cfg["paths"]["json_a_path_val"]
                quesFile = cfg["paths"]["json_q_path_val"]
                vqa = VQA(annFile, quesFile)
                vqaRes = vqa.loadRes(valFile, quesFile)
                vqaEval = VQAEval(vqa, vqaRes, n=2)
                vqaEval.evaluate()
                acc_overall = vqaEval.accuracy['overall']
                # acc_perQuestionType = vqaEval.accuracy['perQuestionType']
                # acc_perAnswerType = vqaEval.accuracy['perAnswerType']
                summary_writer.add_scalar("Acc/overall_" + phase + "_Epoch", acc_overall, global_step=epoch)
                # summary_writer.add_scalar("Acc/perQues" + phase + "_Epoch", epoch_loss, global_step=epoch)
                # summary_writer.add_scalar("Acc/" + phase + "_Epoch", epoch_loss, global_step=epoch)

        # Save the model check points.
        _save_checkpoint(net_name, model, optimizer, epoch, tr_iter, val_iter, lr, cfg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-cfg", "--configuration", default="/home/adnen/Documents/Studium/MScThesis/src/basic_vqa/config/deeQ_dnc_advanced.ini")
    args = vars(arg_parser.parse_args())
    cfg = load_ini_file(args["configuration"])
    main(cfg)
