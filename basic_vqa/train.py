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
from models import VqaModel, VqaModelDncQ, VqaModelDncC
from vqa_eval.PythonHelperTools.vqaTools.vqa import VQA
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


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

    assert not(cfg["hyperparameters"]["use_dnc_c"] and cfg["hyperparameters"]["use_dnc_q"])
    _set_seed(cfg["hyperparameters"]["seed"])
    if cfg["hyperparameters"]["use_dnc_c"]:
        model = VqaModelDncC(cfg).to(device)
        net_name = "dnc_C_" + str(cfg["dnc"]["number"])

    elif cfg["hyperparameters"]["use_dnc_q"]:
        model = VqaModelDncQ(cfg).to(device)
        net_name = "dnc_Q"

    else:
        model = VqaModel(cfg).to(device)
        net_name = "Baseline"

        # embed_size=cfg["hyperparameters"]["commun_embed_size"],
        # qst_vocab_size=qst_vocab_size,
        # ans_vocab_size=ans_vocab_size,
        # word_embed_size=cfg["hyperparameters"]["embedding_dim"],
        # num_layers=args.num_layers,
        # hidden_size=args.hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    if cfg["hyperparameters"]["use_dnc_c"]:
        dnc_params = {"params": model.dnc.parameters(), "lr": cfg["dnc_c"]["lr"]}
        img_encoder_params = {"params": model.img_encoder.fc.parameters()}
        qst_encoder_params = {"params": model.qst_encoder.fc.parameters()}
        if cfg["hyperparameters"]["optimizer"] == "adam":
            optimizer = optim.Adam(
                [dnc_params, img_encoder_params, qst_encoder_params],
                lr=cfg["hyperparameters"]["lr"],
                weight_decay=cfg["hyperparameters"]["weight_decay"])
        elif cfg["hyperparameters"]["optimizer"] == "sgd":
            optimizer = optim.SGD(
                [dnc_params, img_encoder_params, qst_encoder_params],
                lr=cfg["hyperparameters"]["lr"],
                weight_decay=cfg["hyperparameters"]["weight_decay"])
    elif cfg["hyperparameters"]["use_dnc_q"]:
        dnc_params = {"params": model.qst_encoder.dnc_q.parameters(), "lr": cfg["dnc_q"]["lr"]}
        embed_params = {"params": model.qst_encoder.word2vec.parameters()}
        img_encoder_params = {"params": model.img_encoder.fc.parameters()}
        #qst_encoder_params = {"params": model.qst_encoder.fc.parameters()}
        fc1_params = {"params": model.fc1.parameters()}
        fc2_params = {"params": model.fc2.parameters()}

        if cfg["hyperparameters"]["optimizer"] == "adam":
            optimizer = optim.Adam(
                [dnc_params, embed_params, img_encoder_params, fc1_params, fc2_params],
                lr=cfg["hyperparameters"]["lr"],
                weight_decay=cfg["hyperparameters"]["weight_decay"])
        elif cfg["hyperparameters"]["optimizer"] == "sgd":
            optimizer = optim.SGD(
                [dnc_params, embed_params, img_encoder_params, fc1_params, fc2_params],
                lr=cfg["hyperparameters"]["lr"],
                weight_decay=cfg["hyperparameters"]["weight_decay"])
    else:
        params = list(model.img_encoder.fc.parameters()) \
            + list(model.qst_encoder.parameters()) \
            + list(model.fc1.parameters()) \
            + list(model.fc2.parameters())
        optimizer = optim.Adam(params, lr=cfg["hyperparameters"]["lr"])
    print("Training " + net_name)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg["hyperparameters"]["lr_reduce_after"], gamma=cfg["hyperparameters"]["lr_decay_rate"])
    summary_writer = SummaryWriter(logdir=cfg["logging"]["tensorboard_dir"])
    tr_iter = 0
    val_iter = 0
    lr = 0
    lr_dnc = 0

    for epoch in range(cfg["hyperparameters"]["num_epochs"]):

        for phase in ['train', 'valid']:
            if cfg["hyperparameters"]["use_dnc_c"]:
                if cfg["dnc"]["number"] == 1:
                    model.dnc.update_batch_size(cfg["hyperparameters"]["batch_size"])
                    h, mem = model.dnc.reset()
                elif cfg["dnc"]["number"] == 0:
                    (mem, rv) = model.dnc.init_hidden(None, cfg["hyperparameters"]["batch_size"], True)
                else:
                    raise ValueError("No dnc number " + cfg["dnc"]["number"])
            if cfg["hyperparameters"]["use_dnc_q"]:
                (chx, mhx, rv) = (None, None, None)

            running_loss = 0.0
            dataloader = data_loader[phase]
            batch_step_size = len(dataloader.dataset) / cfg["hyperparameters"]["batch_size"]
            if phase == 'train':
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
                    if cfg["hyperparameters"]["use_dnc_c"]:
                        if cfg["dnc"]["number"] == 1:
                            model.dnc.update_batch_size(image.size(0))
                            h, mem = model.dnc.reset()
                        elif cfg["dnc"]["number"] == 0:
                            (mem, rv) = model.dnc.init_hidden(None, image.size(0), False)
                        else:
                            raise ValueError("No dnc number " + cfg["dnc"]["number"])
                    if cfg["hyperparameters"]["use_dnc_q"]:
                        (chx, mhx, rv) = (None, None, None)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if cfg["hyperparameters"]["use_dnc_c"]:
                        if cfg["dnc"]["number"] == 1:
                            output, h, mem = model(image, question, h=h, mem=mem)
                        elif cfg["dnc"]["number"] == 0:
                            output, (mem, rv), v = model(image, question, mem=mem, rv=rv)

                    elif cfg["hyperparameters"]["use_dnc_q"]:
                        output, (chx, mhx, rv), v = model(image, question, chx=chx, mhx=mhx, rv=rv)
                    else:
                        output = model(image, question)      # [batch_size, ans_vocab_size=1000]
                    _, pred = torch.max(output, 1)  # [batch_size]
                    # _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                    loss = criterion(output, label)
                    if phase == 'train':
                        loss.backward()
                        # if iter % cfg["hyperparameters"]["grad_flow_interval"] == 0:
                        #     plot_grad_flow(model.named_parameters(), cfg["hyperparameters"]["grad_flow_dir"], str(tr_iter))
                        if cfg["hyperparameters"]["use_clip_grad"]:
                            nn.utils.clip_grad_norm_(model.parameters(), cfg["hyperparameters"]["clip_value"])
                        optimizer.step()

                        if cfg["hyperparameters"]["use_dnc_c"]:
                            lr_dnc = optimizer.param_groups[0]["lr"]
                            lr = optimizer.param_groups[1]["lr"]
                            dict_lr = {"DNC": lr_dnc, "Rest": lr}
                            summary_writer.add_scalars("lr", dict_lr, global_step=tr_iter)
                        elif cfg["hyperparameters"]["use_dnc_q"]:
                            lr_dnc = optimizer.param_groups[0]["lr"]
                            lr = optimizer.param_groups[1]["lr"]
                            dict_lr = {"DNC": lr_dnc, "Rest": lr}
                            summary_writer.add_scalars("lr", dict_lr, global_step=tr_iter)
                        else:
                            lr = optimizer.param_groups[0]["lr"]
                            summary_writer.add_scalar("lr", lr, global_step=tr_iter)

                    else:
                        question_ids = batch_sample["question_id"].tolist()
                        pred = pred.tolist()
                        pred = [ans_list[i] for i in pred]
                        for id_, ans in zip(question_ids, pred):
                            val_predictions.append({
                                "question_id": id_,
                                "answer": ans
                            })
                    if cfg["hyperparameters"]["use_dnc_c"]:
                        if cfg["dnc"]["number"] == 1:
                            mem = repackage_hidden(mem)
                        elif cfg["dnc"]["number"] == 0:
                            mem = {k: (v.detach() if isinstance(v, var) else v) for k, v in mem.items()}
                            rv = rv.detach()
                    elif cfg["hyperparameters"]["use_dnc_q"]:
                            mhx = {k: (v.detach() if isinstance(v, var) else v) for k, v in mhx.items()}

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                # pred_exp2[pred_exp2 == ans_unk_idx] = -9999
                running_loss += loss.item()
                summary_writer.add_scalar("Loss/" + phase + "_Batch",
                                           loss.item(),
                                           global_step=tr_iter if phase == "train" else val_iter)
                # running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                # running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()

                # Print the average loss in a mini-batch.
                # if batch_idx % 10 == 0:
                #     print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                #           .format(phase.upper(), epoch+1, args.num_epochs, batch_idx, int(batch_step_size), loss.item()))

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


            # epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)      # multiple choice
            # epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)      # multiple choice

            # print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n'
            #       .format(phase.upper(), epoch+1, args.num_epochs, epoch_loss, epoch_acc_exp1, epoch_acc_exp2))

            # Log the loss and accuracy in an epoch.
            # with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt')
            #           .format(phase, epoch+1), 'w') as f:
            #     f.write(str(epoch+1) + '\t'
            #             + str(epoch_loss) + '\t'
            #             + str(epoch_acc_exp1.item()) + '\t'
            #             + str(epoch_acc_exp2.item()))

        # Save the model check points.

        _save_checkpoint(net_name, model, optimizer, epoch, tr_iter, val_iter, lr, cfg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-cfg", "--configuration", default="/home/adnen/Documents/Studium/MScThesis/src/basic_vqa/config/deeQ_dnc_advanced.ini")
    args = vars(arg_parser.parse_args())
    cfg = load_ini_file(args["configuration"])
    main(cfg)
