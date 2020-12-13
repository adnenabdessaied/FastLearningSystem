__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

"""
    Impelementation of some functions needed for logging and saving/loading
    checkpoints.
"""
import os
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.nn as nn
import copy
import numpy as np
import logging
from datetime import datetime
from VQA.preprocessing.reader import Reader
import textdistance
from VisualDialog.preprocessing.statics import (
    DIALOGS,
    ANSWER_OPTS,
    ANSWERS,
    QUESTIONS,
    QUESTION,
    CAPTIONS
    )

logging.basicConfig(level=logging.INFO)


def _unit_simplex_initialization(seed, batch_size, shape):
    mat = []
    for _ in range(batch_size):
        np.random.seed(seed)
        mat.append(list(np.random.uniform(0, 1 / np.sum(shape), shape)))
    return mat


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


@torch.no_grad()
def _weight_init(m):
    if type(m) == nn.Linear:
        if m.weight.requires_grad:
            nn.init.xavier_uniform_(m.weight)
            print(m.weight.data)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.zeros_(m.weight)
    if type(m) == nn.LSTM:
        nn.init.zeros_(m.bias_hh_l0)
        nn.init.zeros_(m.bias_hh_l1)
        nn.init.zeros_(m.bias_ih_l0)
        nn.init.zeros_(m.bias_ih_l1)

        nn.init.xavier_uniform_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_hh_l1)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.xavier_uniform_(m.weight_ih_l1)

    if type(m) == nn.Embedding:
        nn.init.uniform_(m.weight, -1.0, 1.0)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def _init_dnc_states(net, cfg):
    dnc_inputs = []
    if cfg["dnc"]["use_encoder_dnc"]:
        hidden_state_q, memory_state_q = net.encoder.question_dnc.reset()
        hidden_state_h, memory_state_h = net.encoder.history_dnc.reset()
        dnc_inputs.append(hidden_state_q)
        dnc_inputs.append(memory_state_q)
        dnc_inputs.append(hidden_state_h)
        dnc_inputs.append(memory_state_h)

    if cfg["dnc"]["use_decoder_dnc"]:
        hidden_state_a, memory_state_a = net.decoder.answer_opts_dnc.reset()
        dnc_inputs.append(hidden_state_a)
        dnc_inputs.append(memory_state_a)
    return dnc_inputs


def _get_current_timestamp():
    """Returns the current timestamp to be used to name the checkpoint files
     of the model.

    Returns:
        str -- The current timestamp
    """
    current_time = datetime.now()
    current_time = current_time.strftime("%d-%b-%Y_(%H:%M:%S.%f)")
    return current_time


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


def _decorate_tb_image_vqa(path_to_image, question, answer_gt, pred_answer):
    """Decorate an image to be displayed in tensorboard

    Arguments:
        path_to_image {str} -- Path to the image.
        question {str} -- The asked question.
        answer_gt {str} -- Ground-truth answer.
        pred_answer {str} -- Predicted answer.

    Returns:
        np.ndarray -- The decorated image (normalized to [0, 1]).
    """
    # Read the image
    image    = cv2.imread(path_to_image)

    # Convert BGR to RGB as cv2 reads color images in BGR format
    B, R     = copy.deepcopy(image[:, :, 0]), copy.deepcopy(image[:, :, -1])
    image[:, :, 0], image[:, :, -1] = R, B

    # Resize every image to a fixed size (600, 400)
    image    = cv2.resize(image, (600, 600))
    # Join the tokenized question
    question = " ".join(question) + "?"
    # Add a black background where the dialog will be displayed
    black_background = np.zeros((600, 1000, 3), dtype=np.uint8)
    for i, text in enumerate([question, answer_gt, pred_answer]):
        if i % 3 == 0:
            color = (255, 255, 255)
        elif (i - 1) % 3 == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(black_background, text, (10, 31 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    decorated_image = np.concatenate((image, black_background), axis=1)
    return decorated_image / 255.0


def _get_tb_image_grid_visdiag(path_to_image, gts, output, idx, rounds):
    """
    This function randomly takes one image of the batch and displays the gt index and the top predicted indices on top
    of it for each round of the dialog.
    :param path_to_image: The path to the image.
    :param gts: The ground truths of the whole batch.
    :param output: The output of the network for the whole batch.
    :param idx: A randomly chosen index between 0 and batch_size - 1
    :param rounds: A list of rounds we want to visualize.
    :return: A pytorch tensor of decorated images that will be displayed in tensorboard.
    """
    # Read the image
    image         = cv2.imread(path_to_image)

    # Convert BGR to RGB as cv2 reads color images in BGR format
    B, R          = copy.deepcopy(image[:, :, 0]), copy.deepcopy(image[:, :, -1])
    image[:, :, 0], image[:, :, -1] = R, B

    # Resize every image to a fixed size (600, 400)
    image         = cv2.resize(image, (600, 400))

    # Make a copy of the each for each dialog round
    images        = [copy.deepcopy(image) for _ in range(len(rounds))]

    # Extract the gts corresponding to the randomly chosen image
    gts           = gts[idx * 10: (idx + 1) * 10]
    gts           = gts[rounds]
    gts_text      = [str(gt.item()) for gt in gts]
    output        = output.detach()

    # Extract the outputs corresponding to the randomly chosen image
    output        = output[idx * 10: (idx + 1) * 10][:]
    output        = output[rounds][:]

    _, output_idx = output.sort(dim=1, descending=True)
    output_size   = output.size()

    top_1         = [output_idx[i][:1] for i in range(output_size[0])]
    top_1_texts   = ["[{}]".format(str(predicted.item())) for predicted in top_1]

    top_3         = [output_idx[i][:3].tolist() for i in range(output_size[0])]
    top_3_texts   = ["[{}, {}, {}]".format(*predicted) for predicted in top_3]

    top_5         = [output_idx[i][:5].tolist() for i in range(output_size[0])]
    top_5_texts   = ["[{}, {}, {}, {}, {}]".format(*predicted) for predicted in top_5]

    image_grid    = []

    # Decorate each image as described in function doc string.
    for i, (img, gt_text, top_1_text, top_3_text, top_5_text) in enumerate(zip(
            images, gts_text, top_1_texts, top_3_texts, top_5_texts)):

        cv2.putText(img, "Round {}:".format(rounds[i] + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "gt index:" + gt_text, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, "top_1_pred:" + top_1_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, "top_3_pred:" + top_3_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, "top_5_pred:" + top_5_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        image_grid.append(torch.from_numpy(img)/255.0)

    image_grid = torch.stack(image_grid, dim=0)

    return image_grid


def _decorate_tb_image_visdiag(path_to_image, gts, output, idx, best_k_pred, image_id, dialog_reader):

    # Read the image
    image                 = cv2.imread(path_to_image)

    # Convert BGR to RGB as cv2 reads color images in BGR format
    B, R                  = copy.deepcopy(image[:, :, 0]), copy.deepcopy(image[:, :, -1])
    image[:, :, 0], image[:, :, -1] = R, B

    # Resize every image to a fixed size (600, 400)
    image                 = cv2.resize(image, (1000, 1000))

    # Extract the gts corresponding to the randomly chosen image
    gts                   = gts[idx * 10: (idx + 1) * 10]
    dialog                = dialog_reader.data_holders[DIALOGS][image_id]

    raw_questions         = []
    raw_answers           = []
    raw_predicted_answers = []

    caption               = dialog_reader.data_holders[CAPTIONS][image_id]
    caption               = " ".join(caption)
    cv2.putText(image, "C: {}".format(caption), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Detach tensor from graph to avoid memory leaks
    output                = output.detach()

    # Extract the outputs corresponding to the randomly chosen image
    output                = output[idx * 10: (idx + 1) * 10][:]
    _, output_indices     = output.sort(dim=1, descending=True)
    for (diag_round, gt_idx, predicted_idx) in zip(dialog, gts.tolist(), output_indices.tolist()):
        raw_questions.append(" ".join(dialog_reader.data_holders[QUESTIONS][diag_round[QUESTION]]) + "?")
        raw_answers.append(" ".join(dialog_reader.data_holders[ANSWERS][diag_round[ANSWER_OPTS][gt_idx]]))
        raw_predicted_answers.append(" ".join(dialog_reader.data_holders[ANSWERS][
                                                  diag_round[ANSWER_OPTS][predicted_idx[best_k_pred - 1]]]))

    raw_text = []
    for raw_question, raw_predicted_answer, raw_answer in zip(raw_questions, raw_predicted_answers, raw_answers):
        raw_text.append(raw_question)
        raw_text.append(raw_predicted_answer)
        raw_text.append(raw_answer)

    # Add a black background where the dialog will be displayed
    black_background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for i, text in enumerate(raw_text):
        if i % 3 == 0:
            color = (255, 255, 255)
        elif (i - 1) % 3 == 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.putText(black_background, text, (10, 31 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    decorated_image = np.concatenate((image, black_background), axis=1)
    return decorated_image / 255.0


def _set_learning_rate(optimizer, new_lr):
    """ Updates the learning rate of an optimizer.

    Arguments:
        optimizer {torch.optim.Otimizer} -- The optimizer used in training.
        new_lr {float} -- The new learning rate.
    """
    for param in optimizer.param_groups:
        param["lr"] = new_lr


def _compute_extensive_accuracies(question_ids, pred_answers, human_answers_epoch, all_data):
    """Compute all accuracies.

    Arguments:
        question_ids {list} -- List of the question ids.
        pred_answers {list} -- List of the predicted answers.
        human_answers_epoch {list} -- List of the human predicted answers.
        all_data {dict} -- The data of the dataset.

    Returns:
        dict -- A dict containing all the computed accuracies.
    """
    accuracies = {"Per_question_type_accuracy"  : {},
                  "Per_answer_type_accuracy"    : {},
                  "Per_question_length_accuracy": {}}
    frequencies_q  = {}
    frequencies_a  = {}
    frequencies_ql = {}
    for quesdtion_id, pred_answer, human_answers in zip(
        question_ids, pred_answers, human_answers_epoch):
        human_answers = torch.tensor(human_answers)
        data          = all_data[quesdtion_id]
        question_type = data["question_type"]
        answer_type   = data["answer_type"]
        question_len  = str(data["question_len"])
        acc           = (pred_answer == human_answers.cpu()).sum(0).div(3.0)
        acc           = min(acc.item(), 1.0) * 100.0

        if question_type not in accuracies["Per_question_type_accuracy"]:
            frequencies_q[question_type] = 1
            accuracies["Per_question_type_accuracy"][question_type] = acc
        else:
            f = frequencies_q[question_type]
            accuracies["Per_question_type_accuracy"][question_type] = \
                (accuracies["Per_question_type_accuracy"][question_type] * f + acc)/(f + 1)
            frequencies_q[question_type] += 1

        if answer_type not in accuracies["Per_answer_type_accuracy"]:
            frequencies_a[answer_type] = 1
            accuracies["Per_answer_type_accuracy"][answer_type] = acc
        else:
            f = frequencies_a[answer_type]
            accuracies["Per_answer_type_accuracy"][answer_type] = \
                (accuracies["Per_answer_type_accuracy"][answer_type] * f + acc)/(f + 1)
            frequencies_a[answer_type] += 1

        if question_len not in accuracies["Per_question_length_accuracy"]:
            frequencies_ql[question_len] = 1
            accuracies["Per_question_length_accuracy"][question_len] = acc
        else:
            f = frequencies_ql[question_len]
            accuracies["Per_question_length_accuracy"][question_len] = \
                (accuracies["Per_question_length_accuracy"][question_len] * f + acc)/(f + 1)
            frequencies_ql[question_len] += 1

    return accuracies


def MaskedCrossEntropy(logits, target, mask):
    # --> https://github.com/pytorch/pytorch/issues/563
    logp = F.log_softmax(logits, dim=1)
    ymask = logp.data.new(logp.size()).zero_()
    ymask.scatter_(1, target.view(-1, 1), 1)
    ymask = Variable(ymask)
    logpy = (logp * ymask)
    # <--
    # Only keep the answers, i.e. where output = "-"
    logpy = (logpy * mask.unsqueeze(1)).sum(1)
    nlll = -logpy.mean()
    return nlll


def get_predictions(logits, answers_idx):
    logits = logits.detach().cpu()
    arg_max = torch.argmax(torch.softmax(logits, -1), -1)
    arg_max = arg_max.tolist()
    answers_idx = answers_idx.tolist()
    predictions = []
    for idx, o in zip(answers_idx, arg_max):
        pred = [o[i] for i in idx if i >= 0]
        predictions.append(pred)
    return predictions

def get_predictions_visdiag(logits, answers_idx, i2w):
    logits = logits.detach().cpu()
    arg_max = torch.argmax(torch.softmax(logits, -1), -1).squeeze(0)
    arg_max = arg_max.tolist()
    answers_idx = answers_idx.tolist()
    predictions = []
    for idx, o in zip(answers_idx, arg_max):
        pred = [i2w[o[i]] for i in idx if i >= 0]
        predictions.append(pred)
    return predictions


def regroup(inputs, lenghts):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.tolist()
    output = []
    for input_, length in zip(inputs, lenghts):
        o = []
        # assert len(input_) == sum(length)
        for l in length:
            if l <= 0:
                break
            o.append(input_[:l])
            input_ = input_[l:]
        output.append(o)
    return output

def regroup_visdiag(inputs, lenghts):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.tolist()
    output = []
    for input_, length in zip(inputs, lenghts):
        o = []
        # assert len(input_) == sum(length)
        for l in length:
            if l <= 0:
                break
            o.append(" ".join(input_[:l]))
            input_ = input_[l:]
        output.append(o)
    return output


def accuracy(answers_gt, predictions):
    acc_avg, acc = __get_acc(answers_gt, predictions)
    return acc_avg, acc


def __get_acc(gts, preds):
    assert len(gts) == len(preds)
    res = []
    for gt, pred in zip(gts, preds):
        acc = 0
        for g, p in zip(gt, pred):
            if g == p:
                acc += 1
        acc = 100.0 * (acc / len(gt))
        res.append(acc)
    acc_avg = sum(res) / len(res)
    return acc_avg, res


def get_tb_text_babi(answers_gt, predictions, accuracy_list, idx_to_word, n):
    if n > len(answers_gt):
        n = len(answers_gt)
    text = ""
    max_len_gt = 0
    max_len_pred = 0
    max_len_acc = 0
    gt_list = []
    pred_list = []
    acc_list = []
    answers_gt    = answers_gt[:n]
    predictions   = predictions[:n]
    accuracy_list = accuracy_list[:n]
    for gt, pred, acc in zip(answers_gt, predictions, accuracy_list):
        acc_text = str(acc) + "%"
        gt_text = ""
        pred_text = ""

        for g, p in zip(gt, pred):
            for g_, p_ in zip(g, p):
                gt_text   += idx_to_word[g_] + "/"
                pred_text += idx_to_word[p_] + "/"
            gt_text    = gt_text[:-1]
            pred_text  = pred_text[:-1]
            gt_text   += "-"
            pred_text += "-"
        gt_text = gt_text[:-1]
        pred_text = pred_text[:-1]

        gt_list.append(gt_text)
        pred_list.append(pred_text)
        acc_list.append(acc_text)

        if len(gt_text) > max_len_gt:
            max_len_gt = len(gt_text)

        if len(pred_text) > max_len_pred:
            max_len_pred = len(pred_text)

        if len(acc_text) > max_len_acc:
            max_len_acc = len(acc_text)

    max_len_gt   = max(max_len_gt, len("GTS"))
    max_len_pred = max(max_len_pred, len("Predictions"))
    max_len_acc  = max(max_len_acc, len("Accuracy"))

    for gt, pred, acc in zip(gt_list, pred_list, acc_list):
        gt_aligned   = " " * (int(max_len_gt / 2) - int(len(gt) / 2)) + gt + " " * (int(max_len_gt / 2) - int(len(gt) / 2))
        pred_aligned = " " * (int(max_len_pred / 2) - int(len(pred) / 2)) + pred + " " * (int(max_len_pred / 2) - int(len(pred) / 2))
        acc_aligned  = " " * (int(max_len_acc / 2) - int(len(acc) / 2)) + acc + " " * (int(max_len_acc / 2) - int(len(acc) / 2))
        text += "\t" + gt_aligned + "\t" + pred_aligned + "\t" + acc_aligned + "\n"

    header_gts  = " " * (int(max_len_gt / 2) - 1) + "GTS" + " " * (int(max_len_gt / 2) - 1)
    header_pred = " " * (int(max_len_pred / 2) - 5) + "Predictions" + " " * (int(max_len_pred / 2) - 5)
    header_acc  = " " * (int(max_len_acc / 2) - 4) + "Accuracy" + " " * (int(max_len_acc / 2) - 4)
    header = "\t" + header_gts + "\t" + header_pred + "\t" + header_acc + "\n"
    text = header + " "*100 + "\n" + text
    return text


def get_tb_text_visdiag():
    pass

def get_tb_test_summary(acc_dnc_list, acc_lstm_list, tasks):
    max_len_dnc  = max(acc_dnc_list, key=lambda s: len(s))
    max_len_lstm = max(acc_lstm_list, key=lambda s: len(s))
    max_len_task = max(tasks, key=lambda s: len(s))

    max_len_dnc  = max(len(max_len_dnc), len("DNC"))
    max_len_lstm = max(len(max_len_lstm), len("LSTM"))
    max_len_task = max(len(max_len_task), len("Tasks"))

    text = ""
    for task, acc_dnc, acc_lstm in zip(tasks, acc_dnc_list, acc_lstm_list):
        task_aligned     = " " * (int(max_len_task / 2) - int(len(task) / 2)) + task + " " * (int(max_len_task / 2) - int(len(task) / 2))
        acc_dnc_aligned  = " " * (int(max_len_dnc / 2) - int(len(acc_dnc) / 2)) + acc_dnc + " " * (int(max_len_dnc / 2) - int(len(acc_dnc) / 2))
        acc_lstm_aligned = " " * (int(max_len_lstm / 2) - int(len(acc_lstm) / 2)) + acc_lstm + " " * (int(max_len_lstm / 2) - int(len(acc_lstm) / 2))
        text += "\t" + task_aligned + "\t" + acc_dnc_aligned + "\t" + acc_lstm_aligned + "\n"

    header_task = " " * (int(max_len_task / 2) - 2) + "Tasks" + " " * (int(max_len_task / 2) - 2)
    header_dnc  = " " * (int(max_len_dnc / 2) - 1) + "DNC" + " " * (int(max_len_dnc / 2) - 1)
    header_lstm = " " * (int(max_len_task / 2) - 2) + "LSTM" + " " * (int(max_len_task / 2) - 2)
    header = "\t" + header_task + "\t" + header_dnc + "\t" + header_lstm + "\n"
    text = header + " "*100 + "\n" + text
    return text

def get_similarities_visdiag(predictions, opts):
    similarities = []
    for batch_pred, batch_opts in zip(predictions, opts):
        similarities_batch = []
        for round_pred, round_opts in zip(batch_pred, batch_opts):
            similarities_round = [textdistance.levenshtein.normalized_similarity(opts, round_pred) for opts in round_opts]
            similarities_batch.append(similarities_round)
        similarities.append(similarities_batch)
    try:
        similarities = torch.tensor(similarities)
    except Exception as e:
        for i, (batch_pred, batch_opts) in enumerate(zip(predictions, opts)):
            print("-" * 10 + "Batch {}".format(i + 1) + "-" * 10 + "\n")
            for r, (round_pred, round_opts) in enumerate(zip(batch_pred, batch_opts)):
                print("-" * 5 + "Round {}".format(r + 1) + "-" * 5 + "\n")
                print("Prediction = " + round_pred + "\n")
                print("answer_opts = \n")
                for n, o in enumerate(round_opts):
                    print("opt_{}".format(n+1) + "=" + o)
        
    return similarities